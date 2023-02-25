#define PY_SSIZE_T_CLEAN

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <omp.h>
#include "uthash.h"
#include <sys/time.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))

#define DEBUG 0
#define NMAX 65536
#define NEBMAX 1e6

#define HASH_FIND_LONG(head, findlong, out) \
    HASH_FIND(hh, head, findlong, sizeof(long), out)
#define HASH_ADD_LONG(head, longfield, add) \
    HASH_ADD(hh, head, longfield, sizeof(long), add)

typedef struct item_int
{
    int key;
    int val;
    UT_hash_handle hh;
} dict_int;

typedef struct item_long
{
    long key;
    int val;
    UT_hash_handle hh;
} dict_long;

/* hash of hashes */
typedef struct item
{
    int key;
    int val;
    struct item *sub;
    UT_hash_handle hh;
} dict_item;

void add_item(dict_int **maps, int key)
{
    dict_int *s;
    HASH_FIND_INT(*maps, &key, s); /* s: output pointer */
    if (s == NULL)
    {
        dict_int *k = malloc(sizeof(*k));
        k->key = key;
        HASH_ADD_INT(*maps, key, k);
    }
}

static int find_key_int(dict_int *maps, int key)
{
    dict_int *s;
    HASH_FIND_INT(maps, &key, s); /* s: output pointer */
    return s ? s->val : -1;
}

static int find_key_long(dict_long *maps, long key)
{
    dict_long *s;
    HASH_FIND_LONG(maps, &key, s); /* s: output pointer */
    return s ? s->val : -1;
}

static int find_key_item(dict_item *items, int key)
{
    dict_item *s;
    HASH_FIND_INT(items, &key, s); /* s: output pointer */
    return s ? s->val : -1;
}

static int find_idx(dict_item *items, int key1, int key2)
{
    dict_item *s, *p;
    HASH_FIND_INT(items, &key1, s); /* s: output pointer */
    if (s != NULL)
    {
        HASH_FIND_INT(s->sub, &key2, p);
        return p ? p->val : 0;
    }
    else
    {
        return -1;
    }
}

void delete_all(dict_item *items)
{
    dict_item *item1, *item2, *tmp1, *tmp2;

    /* clean up two-level hash tables */
    HASH_ITER(hh, items, item1, tmp1)
    {
        HASH_ITER(hh, item1->sub, item2, tmp2)
        {
            HASH_DEL(item1->sub, item2);
            free(item2);
        }
        HASH_DEL(items, item1);
        free(item1);
    }
}

// demo func
static PyObject *adds(PyObject *self, PyObject *args)
{
    int arg1, arg2;
    if (!(PyArg_ParseTuple(args, "ii", &arg1, &arg2)))
    {
        return NULL;
    }
    return Py_BuildValue("i", arg1 * 2 + arg2 * 7);
}

static PyObject *exec(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

// helper func
static void f_format(const npy_intp *dims, int *CArrays)
{
    for (int x = 0; x < dims[0]; x++)
    {
        printf("idx %d: ", x);
        for (int y = 0; y < dims[1]; y++)
        {
            printf("%d ", CArrays[x * dims[1] + y]);
        }
        printf("\n");
    }
}

static void random_walk(int const *ptr, int const *neighs, int const *seq, int n, int num_walks, int num_steps, int seed, int nthread, int *walks)
{
    /* https://github.com/lkskstlr/rwalk */
    if (DEBUG)
    {
        printf("[Input] n: %d, num_walks: %d, num_steps: %d, seed: %d, nthread: %d\n", n, num_walks, num_steps, seed, nthread);
    }
    if (nthread > 0)
    {
        omp_set_num_threads(nthread);
    }
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        unsigned int private_seed = (unsigned int)(seed + thread_num);
#pragma omp for
        for (int i = 0; i < n; i++)
        {
            int offset, num_neighs;
            for (int walk = 0; walk < num_walks; walk++)
            {
                int curr = seq[i];
                offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
                walks[offset] = curr;
                for (int step = 0; step < num_steps; step++)
                {
                    num_neighs = ptr[curr + 1] - ptr[curr];
                    if (num_neighs > 0)
                    {
                        curr = neighs[ptr[curr] + (rand_r(&private_seed) % num_neighs)];
                    }
                    walks[offset + step + 1] = curr;
                }
            }
        }
    }
}

// random walk without replacement (1st neigh)
static void random_walk_wo(int const *ptr, int const *neighs, int const *seq, int n, int num_walks, int num_steps, int seed, int nthread, int *walks)
{
    if (nthread > 0)
    {
        omp_set_num_threads(nthread);
    }
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        unsigned int private_seed = (unsigned int)(seed + thread_num);

#pragma omp for
        for (int i = 0; i < n; i++)
        {
            int offset, num_neighs;

            int num_hop1 = ptr[seq[i] + 1] - ptr[seq[i]];
            int rseq[num_hop1];
            if (num_hop1 > num_walks)
            {
                // https://www.programmersought.com/article/71554044511/
                int s, t;
                for (int j = 0; j < num_hop1; j++)
                    rseq[j] = j;
                for (int k = 0; k < num_walks; k++)
                {
                    s = rand_r(&private_seed) % (num_hop1 - k) + k;
                    t = rseq[k];
                    rseq[k] = rseq[s];
                    rseq[s] = t;
                }
            }

            for (int walk = 0; walk < num_walks; walk++)
            {
                int curr = seq[i];
                offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
                walks[offset] = curr;
                if (num_hop1 < 1)
                {
                    walks[offset + 1] = curr;
                }
                else if (num_hop1 <= num_walks)
                {
                    curr = neighs[ptr[curr] + walk % num_hop1];
                    walks[offset + 1] = curr;
                }
                else
                {
                    curr = neighs[ptr[curr] + rseq[walk]];
                    walks[offset + 1] = curr;
                }
                for (int step = 1; step < num_steps; step++)
                {
                    num_neighs = ptr[curr + 1] - ptr[curr];
                    if (num_neighs > 0)
                    {
                        curr = neighs[ptr[curr] + (rand_r(&private_seed) % num_neighs)];
                    }
                    walks[offset + step + 1] = curr;
                }
            }
        }
    }
}

void rpe_encoder(int const *arr, int idx, int num_walks, int num_steps, PyArrayObject **out)
{
    PyArrayObject *oarr1 = NULL, *oarr2 = NULL;
    dict_int *mapping = NULL;
    int offset = idx * num_walks * (num_steps + 1);

    // add the root node
    dict_int *root = malloc(sizeof(*root));
    root->key = arr[offset];
    root->val = 0;
    HASH_ADD_INT(mapping, key, root);

    // add the rest unique nodes
    int count = 1;
    for (int i = 1; i < num_steps + 1; i++)
    {
        for (int j = 0; j < num_walks; j++)
        {
            int token = arr[offset + j * (num_steps + 1) + i];
            if (find_key_int(mapping, token) < 0)
            {
                dict_int *k = malloc(sizeof(*k));
                // walk starts from each node (main key)
                k->key = token;
                k->val = count;
                HASH_ADD_INT(mapping, key, k);
                count++;
            }
        }
    }

    // create an array for unique encoding
    npy_intp odims1[2] = {count, num_steps + 1};
    oarr1 = (PyArrayObject *)PyArray_ZEROS(2, odims1, NPY_INT, 0);
    if (!oarr1)
        PyErr_SetString(PyExc_TypeError, "Failed to initialize a PyArrayObject.");
    int *Coarr1 = (int *)PyArray_DATA(oarr1);

    // create an array for corresponding node id
    npy_intp odims2[1] = {count};
    oarr2 = (PyArrayObject *)PyArray_SimpleNew(1, odims2, NPY_INT);
    if (!oarr2)
        PyErr_SetString(PyExc_TypeError, "Failed to initialize a PyArrayObject.");
    int *Coarr2 = (int *)PyArray_DATA(oarr2);

    Coarr1[0] = num_walks;

    for (int i = 1; i < num_steps + 1; i++)
    {
        for (int j = 0; j < num_walks; j++)
        {
            int anchor = find_key_int(mapping, arr[offset + j * (num_steps + 1) + i]);
            Coarr1[anchor * (num_steps + 1) + i]++;
        }
    }

    // free memory
    dict_int *cur_item, *tmp;
    HASH_ITER(hh, mapping, cur_item, tmp)
    {
        Coarr2[cur_item->val] = cur_item->key;
        HASH_DEL(mapping, cur_item); /* delete it (users advances to next) */
        free(cur_item);              /* free it */
    }
    out[2 * idx] = oarr2, out[2 * idx + 1] = oarr1;
}

static PyObject *walk_sampler(PyObject *self, PyObject *args, PyObject *kws)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *query = NULL;
    PyArrayObject *ptr = NULL, *neighs = NULL, *seq = NULL, *oarr = NULL, *obj_arr = NULL;
    int num_walks = 100, num_steps = 3, seed = 111413, nthread = -1, re = -1;

    static char *kwlist[] = {"ptr", "neighs", "query", "num_walks", "num_steps", "nthread", "seed", "replacement", NULL};
    if (!(PyArg_ParseTupleAndKeywords(args, kws, "OOO|iiiip", kwlist, &arg1, &arg2, &query, &num_walks, &num_steps, &nthread, &seed, &re)))
    {
        PyErr_SetString(PyExc_TypeError, "Input parsing error.\n");
        return NULL;
    }

    /* handle the adjacency matrix of input graph in CSR format */
    ptr = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!ptr)
        return NULL;
    int *Cptr = PyArray_DATA(ptr);

    neighs = (PyArrayObject *)PyArray_FROM_OTF(arg2, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!neighs)
        return NULL;
    int *Cneighs = PyArray_DATA(neighs);

    seq = (PyArrayObject *)PyArray_FROM_OTF(query, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!seq)
        return NULL;
    int *Cseq = PyArray_DATA(seq);

    int n = (int)PyArray_SIZE(seq);

    // create an array for sampled walks
    npy_intp odims[2] = {n, num_walks * (num_steps + 1)};
    oarr = (PyArrayObject *)PyArray_SimpleNew(2, odims, NPY_INT);
    if (oarr == NULL)
        goto fail;
    int *Coarr = (int *)PyArray_DATA(oarr);

    if (re > 0)
    {
        // Using no replacement sampling for the 1-hop
        random_walk_wo(Cptr, Cneighs, Cseq, n, num_walks, num_steps, seed, nthread, Coarr);
    }
    else
    {
        random_walk(Cptr, Cneighs, Cseq, n, num_walks, num_steps, seed, nthread, Coarr);
    }

    // create an object array for rpe and the corresponding node id
    npy_intp obj_dims[2] = {n, 2};
    obj_arr = (PyArrayObject *)PyArray_SimpleNew(2, obj_dims, NPY_OBJECT);
    if (obj_arr == NULL)
        goto fail;
    PyArrayObject **Cobj_arr = PyArray_DATA(obj_arr);

#pragma omp for
    for (int k = 0; k < n; k++)
    {
        rpe_encoder(Coarr, k, num_walks, num_steps, Cobj_arr);
    }

    Py_DECREF(ptr);
    Py_DECREF(neighs);
    Py_DECREF(seq);
    return Py_BuildValue("[N,N]", PyArray_Return(oarr), PyArray_Return(obj_arr));

fail:
    Py_XDECREF(ptr);
    Py_XDECREF(neighs);
    Py_XDECREF(seq);
    PyArray_XDECREF(oarr);
    PyArray_XDECREF(obj_arr);
    return NULL;
}

static PyObject *batch_sampler(PyObject *self, PyObject *args, PyObject *kws)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *query = NULL;
    PyArrayObject *ptr = NULL, *neighs = NULL, *seq = NULL, *oarr = NULL;
    int num_walks = 200, num_steps = 8, seed = 111413, nthread = -1, thld = 1000;

    static char *kwlist[] = {"ptr", "neighs", "query", "num_walks", "num_steps", "thld", "nthread", "seed", NULL};
    if (!(PyArg_ParseTupleAndKeywords(args, kws, "OOO|iiiii", kwlist, &arg1, &arg2, &query, &num_walks, &num_steps, &thld, &nthread, &seed)))
    {
        PyErr_SetString(PyExc_TypeError, "Input parsing error.\n");
        return NULL;
    }

    /* handle the adjacency matrix of input graph in CSR format */
    ptr = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!ptr)
        return NULL;
    int *Cptr = PyArray_DATA(ptr);

    neighs = (PyArrayObject *)PyArray_FROM_OTF(arg2, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!neighs)
        return NULL;
    int *Cneighs = PyArray_DATA(neighs);

    seq = (PyArrayObject *)PyArray_FROM_OTF(query, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!seq)
        return NULL;
    int *Cseq = PyArray_DATA(seq);

    int n = (int)PyArray_SIZE(seq);
    unsigned int private_seed = (unsigned int)(seed + getpid());

    /* initialize the hashtable */
    dict_int *batch = NULL;
    for (int i = 0; i < n; i++)
    {
        int num_hop1 = Cptr[Cseq[i] + 1] - Cptr[Cseq[i]];
        int num_neighs, rseq[num_hop1];

        if (num_hop1 > num_walks)
        {
            int s, t;
            for (int j = 0; j < num_hop1; j++)
                rseq[j] = j;
            for (int k = 0; k < num_walks; k++)
            {
                s = rand_r(&private_seed) % (num_hop1 - k) + k;
                t = rseq[k];
                rseq[k] = rseq[s];
                rseq[s] = t;
            }
        }

        add_item(&batch, Cseq[i]);

        for (int walk = 0; walk < num_walks; walk++)
        {
            int curr = Cseq[i];
            if (num_hop1 < 1)
            {
                break;
            }
            else if (num_hop1 <= num_walks)
            {
                curr = Cneighs[Cptr[curr] + walk % num_hop1];
            }
            else
            {
                curr = Cneighs[Cptr[curr] + rseq[walk]];
            }
            add_item(&batch, curr);

            for (int step = 1; step < num_steps; step++)
            {
                num_neighs = Cptr[curr + 1] - Cptr[curr];
                if (num_neighs > 0)
                {
                    curr = Cneighs[Cptr[curr] + (rand_r(&private_seed) % num_neighs)];
                    add_item(&batch, curr);
                }
            }

            if ((int)HASH_COUNT(batch) >= ((i + 1) * thld / n))
                break;
        }
    }

    // create an array for sampled batch
    npy_intp odims[1] = {HASH_COUNT(batch)};
    oarr = (PyArrayObject *)PyArray_SimpleNew(1, odims, NPY_INT);
    if (oarr == NULL)
        goto fail;
    int *Coarr = (int *)PyArray_DATA(oarr);

    // free memory
    dict_int *cur_item, *tmp;
    int idx = 0;
    HASH_ITER(hh, batch, cur_item, tmp)
    {
        Coarr[idx] = cur_item->key;
        HASH_DEL(batch, cur_item); /* delete it (users advances to next) */
        free(cur_item);            /* free it */
        idx++;
    }

    Py_DECREF(ptr);
    Py_DECREF(neighs);
    Py_DECREF(seq);
    return PyArray_Return(oarr);

fail:
    Py_XDECREF(ptr);
    Py_XDECREF(neighs);
    Py_XDECREF(seq);
    PyArray_XDECREF(oarr);
    return NULL;
}

static PyObject *walk_join(PyObject *self, PyObject *args, PyObject *kws)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *query = NULL, *seq = NULL, **src;
    PyArrayObject *warr = NULL, *qarr = NULL, *oarr = NULL, *xarr = NULL;
    int nthread = -1, req = -1;

    static char *kwlist[] = {"walk", "key", "query", "nthread", "return_idx", NULL};
    if (!(PyArg_ParseTupleAndKeywords(args, kws, "OOO|ip", kwlist, &arg1, &arg2, &query, &nthread, &req)))
    {
        PyErr_SetString(PyExc_TypeError, "Input parsing error.\n");
        return NULL;
    }

    /* handle input walks (numpy array format) */
    warr = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!warr)
        return NULL;
    int *Cwarr = (int *)PyArray_DATA(warr);

    npy_intp *warr_dims = PyArray_DIMS(warr);
    int stride = (PyArray_NDIM(warr) > 2) ? (int)(warr_dims[1] * warr_dims[2]) : (int)warr_dims[1];

    /* handle input keys (a iterable sequence of tuples) */
    seq = PySequence_Fast(arg2, "Argument must be iterable.\n");
    if (!seq)
        return NULL;
    if (PySequence_Fast_GET_SIZE(seq) != warr_dims[0])
    {
        PyErr_SetString(PyExc_AssertionError, "Dims do not match between num of walks and keys.\n");
        return NULL;
    }

    /* handle input queries (numpy array/sequence) */
    qarr = (PyArrayObject *)PyArray_FROM_OTF(query, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!qarr)
        return NULL;
    int *Cqarr = (int *)PyArray_DATA(qarr);

    npy_intp *qarr_dims = PyArray_DIMS(qarr);
    if (DEBUG)
    {
        printf("[Input] dims of query: %d, %d\n", (int)qarr_dims[0], (int)qarr_dims[1]);
    }

    /* initialize the hashtable */
    dict_item *items = NULL;
    int idx = 1;
    src = PySequence_Fast_ITEMS(seq);

    /* build two level hash table: 1) reindex main keys 2) hash unique node idx associated with each key */
    for (int i = 0; i < warr_dims[0]; i++)
    {
        /* make initial element */
        dict_item *k = malloc(sizeof(*k));
        // walk starts from each node (main key)
        k->key = Cwarr[i * stride];
        k->sub = NULL;
        k->val = i;
        HASH_ADD_INT(items, key, k);

        PyObject *item = PySequence_Fast(src[i], "Argument must be iterable.\n");
        int item_size = (!PyArray_CheckExact(item)) ? PySequence_Fast_GET_SIZE(item) : PyArray_Size(item);

        for (int j = 0; j < item_size; j++)
        {
            /* add a sub hash table off this element */
            dict_item *w = malloc(sizeof(*w));
            w->key = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(item, j));
            w->sub = NULL;
            w->val = idx;
            HASH_ADD_INT(k->sub, key, w);
            idx++;
        }

        // must add, to avoid memory leakage
        Py_DECREF(item);
    }

    // create an array for output of remapped index
    npy_intp odims[2] = {2, qarr_dims[0] * 2 * stride};
    oarr = (PyArrayObject *)PyArray_SimpleNew(2, odims, NPY_INT);
    if (oarr == NULL)
        goto fail;
    int *Coarr = (int *)PyArray_DATA(oarr);

    if (DEBUG)
    {
        printf("[Output] dims of array: %d, %d\n", (int)odims[0], (int)odims[1]);
    }

    // create an array for remapped query
    xarr = (PyArrayObject *)PyArray_SimpleNew(2, qarr_dims, NPY_INT);
    if (xarr == NULL)
        goto fail;
    int *Cxarr = (int *)PyArray_DATA(xarr);

    if (nthread > 0)
    {
        omp_set_num_threads(nthread);
    }

#pragma omp parallel for
    for (int x = 0; x < qarr_dims[0]; x++)
    {
        int qid = 2 * x;
        int key1 = Cqarr[qid], key2 = Cqarr[qid + 1];
        Cxarr[qid] = find_key_item(items, key1), Cxarr[qid + 1] = find_key_item(items, key2);
        for (int y = 0; y < 2 * stride; y += 2)
        {
            int offset1 = Cxarr[qid] * stride + y / 2, offset2 = Cxarr[qid + 1] * stride + y / 2;
            Coarr[qid * stride + y] = find_idx(items, key1, Cwarr[offset1]);
            Coarr[qid * stride + y + 1] = find_idx(items, key2, Cwarr[offset1]);
            Coarr[odims[1] + qid * stride + y] = find_idx(items, key1, Cwarr[offset2]);
            Coarr[odims[1] + qid * stride + y + 1] = find_idx(items, key2, Cwarr[offset2]);
        }
    }

    Py_DECREF(warr);
    Py_DECREF(qarr);
    Py_DECREF(seq);
    delete_all(items);
    if (req > 0)
    {
        return Py_BuildValue("[N,N]", PyArray_Return(oarr), PyArray_Return(xarr));
    }
    else
    {
        return PyArray_Return(oarr);
    }

fail:
    Py_XDECREF(warr);
    Py_XDECREF(qarr);
    Py_XDECREF(seq);
    delete_all(items);
    PyArray_XDECREF(oarr);
    PyArray_XDECREF(xarr);
    return NULL;
}

static PyObject *set_sampler(PyObject *self, PyObject *args, PyObject *kws)
{
    PyObject *arg1 = NULL, *arg2 = NULL, *query = NULL;
    PyArrayObject *ptr = NULL, *neighs = NULL, *seq = NULL, *oarr = NULL, *xarr = NULL, *narr = NULL, *earr = NULL;
    int num_walks = 100, num_steps = 3, seed = 111413, nthread = -1, bucket = -1, reduce = -1;

    static char *kwlist[] = {"indptr", "indices", "query", "num_walks", "num_steps", "bucket", "nthread", "seed", "debug", NULL};
    if (!(PyArg_ParseTupleAndKeywords(args, kws, "OOO|iiiiii", kwlist, &arg1, &arg2, &query, &num_walks, &num_steps, &bucket, &nthread, &seed, &reduce)))
    {
        PyErr_SetString(PyExc_TypeError, "Input parsing error.\n");
        return NULL;
    }

    /* handle sparse matrix of adj */
    ptr = (PyArrayObject *)PyArray_FROM_OTF(arg1, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!ptr)
        return NULL;
    int *Cptr = PyArray_DATA(ptr);

    neighs = (PyArrayObject *)PyArray_FROM_OTF(arg2, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!neighs)
        return NULL;
    int *Cneighs = PyArray_DATA(neighs);

    seq = (PyArrayObject *)PyArray_FROM_OTF(query, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!seq)
        return NULL;
    int *Cseq = PyArray_DATA(seq);

    int n = (int)PyArray_SIZE(seq);
    int ncol = num_steps + 1;
    int stride = (bucket < 0) ? num_walks * num_steps + 1 : bucket;

    // create an array for compressed encoding / buffer
    int buffer_size = min(n, NMAX) * stride;
    npy_intp xdims[2] = {buffer_size, ncol};
    xarr = (PyArrayObject *)PyArray_SimpleNew(2, xdims, NPY_SHORT);
    if (xarr == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for buffer.\n");
        goto fail;
    }
    short *buffer = (short *)PyArray_DATA(xarr);

    // Dynamically allocate memory using malloc() for large n
    short *encoding;
    long enc_size = (long)n * num_walks * ncol;
    encoding = (short *)PyMem_RawMalloc(enc_size * sizeof(short));
    if (encoding == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for encoding.\n");
        return NULL;
    }

    // create an array for storing remapped index of encoding
    npy_intp odims[1] = {(long)n * stride};
    oarr = (PyArrayObject *)PyArray_EMPTY(1, odims, NPY_INT, 0);
    int *nidx = (int *)PyArray_DATA(oarr);
    if (oarr == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for indexing.\n");
        goto fail;
    }

    // create an array for storing set size
    npy_intp ndims[1] = {n};
    narr = (PyArrayObject *)PyArray_ZEROS(1, ndims, NPY_INT, 0);
    if (narr == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for storing set size.\n");
        goto fail;
    }
    int *nsize = (int *)PyArray_DATA(narr);

    long *ncumsum;
    ncumsum = (long *)PyMem_RawMalloc((n + 1) * sizeof(long));
    ncumsum[0] = 0;

    if (nthread > 0)
    {
        omp_set_num_threads(nthread);
    }
    int thread_num = omp_get_thread_num();
    uint private_seed = (uint)(seed + thread_num);

    struct timeval wtic, wtac;
    gettimeofday(&wtic, 0);
    int maxset = 0, blk = 1, nblk = 2 + ((n - 1) / NMAX);
    while (blk < nblk)
    {
        memset(buffer, 0, buffer_size * ncol * sizeof(*buffer));
        // max node id 2,147,483,647
        int begin = (blk - 1) * NMAX, end = min(blk * NMAX, n);
#pragma omp parallel
        {
#pragma omp for
            for (int i = begin; i < end; i++)
            {
                dict_int *node_set = NULL;
                int offset = (i % NMAX) * stride * ncol;
                // handle potential very large neighborhood case
                int num_hop1 = min(Cptr[Cseq[i] + 1] - Cptr[Cseq[i]], NEBMAX);
                buffer[offset] = num_walks;

                if (num_hop1 == 0)
                {
                    nsize[i] = 1;
                    for (int step = 1; step <= num_steps; step++)
                    {
                        buffer[offset + step] = num_walks;
                    }
                    continue;
                }

                int num_neighs, rseq[num_hop1];
                if (num_hop1 > num_walks)
                {
                    int s, t;
                    for (int j = 0; j < num_hop1; j++)
                        rseq[j] = j;
                    for (int k = 0; k < num_walks; k++)
                    {
                        s = rand_r(&private_seed) % (num_hop1 - k) + k;
                        t = rseq[k];
                        rseq[k] = rseq[s];
                        rseq[s] = t;
                    }
                }

                // add the root node
                dict_int *root = malloc(sizeof(*root));
                root->key = Cseq[i];
                root->val = 0;
                HASH_ADD_INT(node_set, key, root);

                int count = 1, flag = 0;
                for (int walk = 0; walk < num_walks; walk++)
                {
                    int curr = Cseq[i];
                    for (int step = 0; step < num_steps; step++)
                    {
                        if (step < 1)
                        {
                            // step 0 without replacement
                            if (num_hop1 <= num_walks)
                            {
                                curr = Cneighs[Cptr[curr] + walk % num_hop1];
                            }
                            else
                            {
                                curr = Cneighs[Cptr[curr] + rseq[walk]];
                            }
                        }
                        else
                        {
                            num_neighs = Cptr[curr + 1] - Cptr[curr];
                            if (num_neighs > 0)
                            {
                                curr = Cneighs[Cptr[curr] + (rand_r(&private_seed) % num_neighs)];
                            }
                        }

                        int idx = find_key_int(node_set, curr);
                        if (idx < 0)
                        {
                            if (count < stride)
                            {
                                // unique node visited by walks from the root
                                dict_int *k = malloc(sizeof(*k));
                                k->key = curr;
                                k->val = count;
                                HASH_ADD_INT(node_set, key, k);
                                idx = count;
                                count++;
                            }
                            else
                            {
                                flag = 1;
                                continue;
                            }
                        }
                        buffer[offset + idx * ncol + step + 1]++;
                    }
                }

                nsize[i] = count;
                if (flag)
                    printf("#SubGAcc: key %d exceeds the buffer, try a larger bucket size > %d.\n", Cseq[i], stride);
                // free memory
                dict_int *cur_item, *tmp;
                HASH_ITER(hh, node_set, cur_item, tmp)
                { // store node id to nidx
                    nidx[(long)i * stride + cur_item->val] = cur_item->key;
                    HASH_DEL(node_set, cur_item);
                    free(cur_item);
                }
            }
        }

        for (int j = begin; j < end; j++)
        {
            ncumsum[j + 1] = ncumsum[j] + nsize[j];
            if (nsize[j] > maxset)
            {
                maxset = nsize[j];
            }
            // make sure access enc array in boundary
            while (ncumsum[j + 1] * ncol > enc_size)
            {
                enc_size *= 1.5;
                encoding = PyMem_RawRealloc(encoding, enc_size * sizeof(*encoding));
                if (encoding != NULL)
                {
                    printf("#SubGAcc: resizing enc array to %ld.\n", enc_size / ncol);
                }
                else
                {
                    PyErr_SetString(PyExc_MemoryError, "Failed to reallocate memory of encoding array.\n");
                    return NULL;
                }
            }
            memcpy(encoding + ncumsum[j] * ncol, buffer + (j % NMAX) * stride * ncol, nsize[j] * ncol * sizeof(*buffer));
            memmove(nidx + ncumsum[j], nidx + (long)j * stride, nsize[j] * sizeof(*nidx));
        }
        blk += 1;
    }
    gettimeofday(&wtac, 0);
    double dtw = (long)(wtac.tv_sec - wtic.tv_sec) + (wtac.tv_usec - wtic.tv_usec) * 1e-6;
    long ntotal = ncumsum[n];
    printf("#SubGAcc: #total %ld; #max_set %d of %d; buffer usage %.2f%%; dT_w %.2fs\n", ntotal, maxset, stride, (double)ntotal / n / stride * 100, dtw);

    gettimeofday(&wtic, 0);
    encoding = PyMem_RawRealloc(encoding, ntotal * ncol * sizeof(*encoding));
    if (encoding == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to resize encoding array.\n");
        return NULL;
    }

    npy_intp new_odims[2] = {2, ntotal};
    PyArray_Dims oadims;
    oadims.ptr = new_odims;
    oadims.len = 2;
    PyObject *OarrObj = PyArray_Resize((PyArrayObject *)oarr, &oadims, 0, NPY_ANYORDER);
    if (OarrObj == NULL)
    {
        goto fail;
    }
    Py_DECREF(OarrObj);
    nidx = (int *)PyArray_DATA(oarr);

    ulong LEAD;
    // __builtin_clz returns the number of leading 0-bits in x, starting at the most significant bit position.
    // If x is 0, the result is undefined
    const int SHIFT = 32 - __builtin_clz(num_walks);
    LEAD = (ncol - 1) * SHIFT;
    if (sizeof(ulong) * 8 >= 1 + LEAD)
    {
        printf("#SubGAcc: SHIFT %d, LEAD %ld, w_enc %ld, w_hash %ld ", SHIFT, LEAD, sizeof(*encoding) * 8, sizeof(ulong) * 8);
        LEAD = (ulong)1 << LEAD;
        printf("LEAD mask 0x%016lx\n", LEAD);
    }
    else
    {
        PyErr_SetString(PyExc_AssertionError, "Longer width of type for hasing key needed > INT64.\n");
        return NULL;
    }

    ulong *bithash;
    bithash = (ulong *)PyMem_RawCalloc(ntotal, sizeof(ulong));
    if (bithash == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for hashing.\n");
        return NULL;
    }

    if (reduce > 0)
    {
        // keep a copy of enc for debug
        npy_intp edims[2] = {ntotal, ncol};
        earr = (PyArrayObject *)PyArray_SimpleNew(2, edims, NPY_SHORT);
        int *Cearr = (int *)PyArray_DATA(earr);
        if (earr == NULL)
            goto fail;
        memcpy(Cearr, encoding, ntotal * ncol * sizeof(*encoding));
    }

#pragma omp parallel
    {
#pragma omp for
        // remap encoding to integer
        for (long i = 0; i < ntotal; i++)
        {
            for (int j = 1; j < ncol; j++)
            {
                // int -> bit | concat | shift
                bithash[i] = bithash[i] << SHIFT;
                bithash[i] |= encoding[i * ncol + j];
            }
        }
    }

    // correct landing counts for root nodes
    for (int k = 0; k < n; k++)
    {
        bithash[ncumsum[k]] |= LEAD;
    }

    dict_long *unique = NULL;
    int count = 0, idx;
    ulong curr;
    for (long i = 0; i < ntotal; i++)
    {
        curr = bithash[i];
        idx = find_key_long(unique, curr);
        if (idx < 0)
        {
            dict_long *kh = malloc(sizeof(*kh));
            kh->key = curr;
            kh->val = count;
            HASH_ADD_LONG(unique, key, kh);
            idx = count;
            if ((long)idx != i)
            {
                memmove(encoding + idx * ncol, encoding + i * ncol, ncol * sizeof(*encoding));
            }
            count++;
        }
        nidx[ntotal + i] = idx;
    }
    PyMem_RawFree(ncumsum);
    PyMem_RawFree(bithash);

    npy_intp new_xdims[2] = {count, ncol};
    PyArray_Dims xadims;
    xadims.ptr = new_xdims;
    xadims.len = PyArray_NDIM(oarr);
    PyObject *XarrObj = PyArray_Resize((PyArrayObject *)xarr, &xadims, 0, NPY_ANYORDER);
    if (XarrObj == NULL)
    {
        goto fail;
    }
    Py_DECREF(XarrObj);

    dict_long *cur_item, *tmp;
    HASH_ITER(hh, unique, cur_item, tmp)
    {
        memcpy(buffer + cur_item->val * ncol, encoding + cur_item->val * ncol, ncol * sizeof(*encoding));
        HASH_DEL(unique, cur_item);
        free(cur_item);
        count--;
    }
    if (count != 0)
    {
        PyErr_SetString(PyExc_AssertionError, "Encoding data are corrupted.\n");
        return NULL;
    }
    PyMem_RawFree(encoding);
    gettimeofday(&wtac, 0);
    dtw = (long)(wtac.tv_sec - wtic.tv_sec) + (wtac.tv_usec - wtic.tv_usec) * 1e-6;
    printf("#SubGAcc: #enc_unique %d; compression ratio %.2f, dT_e %.2fs\n", (int)new_xdims[0], ntotal / (float)new_xdims[0], dtw);

    Py_DECREF(ptr);
    Py_DECREF(neighs);
    Py_DECREF(seq);
    if (DEBUG)
        f_format(xdims, (int *)buffer);

    if (reduce > 0)
    {
        return Py_BuildValue("[N,N,N,N]", PyArray_Return(narr), PyArray_Return(oarr), PyArray_Return(xarr), PyArray_Return(earr));
    }
    else
    {
        return Py_BuildValue("[N,N,N]", PyArray_Return(narr), PyArray_Return(oarr), PyArray_Return(xarr));
    }
fail:
    Py_XDECREF(ptr);
    Py_XDECREF(neighs);
    Py_XDECREF(seq);
    PyArray_XDECREF(narr);
    PyArray_XDECREF(oarr);
    PyArray_XDECREF(xarr);
    PyArray_XDECREF(earr);
    return NULL;
}

static PyMethodDef SubGAccMethods[] = {
    {"add", adds, METH_VARARGS, "Addition operation."},
    {"run", exec, METH_VARARGS, "Execute a shell command."},
    {"batch_sampler", (PyCFunction)batch_sampler, METH_VARARGS | METH_KEYWORDS, "Run walk-based sampling for training queries in batches."},
    {"walk_sampler", (PyCFunction)walk_sampler, METH_VARARGS | METH_KEYWORDS, "Run random walks with relative positional encoding (RPE)."},
    {"walk_join", (PyCFunction)walk_join, METH_VARARGS | METH_KEYWORDS, "Online subgraph (walk + RPE) joining for an iterable sequence of queries."},
    {"gset_sampler", (PyCFunction)set_sampler, METH_VARARGS | METH_KEYWORDS, "Run walk-based node set sampling with LP structure encoder."},
    {NULL, NULL, 0, NULL}};

static char subgacc_doc[] = "SubGACC is an extension library based on C and openmp for accelerating subgraph operations.";

static struct PyModuleDef subgacc_module = {
    PyModuleDef_HEAD_INIT,
    "subg_acc",  /* name of module */
    subgacc_doc, /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    SubGAccMethods};

PyMODINIT_FUNC PyInit_subg_acc(void)
{
    import_array();
    return PyModule_Create(&subgacc_module);
}
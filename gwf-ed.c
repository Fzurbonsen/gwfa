#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "gwfa.h"
#include "kalloc.h"
#include "ksort.h"

/*******************
 * Compiler Macros *
 *******************/

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#define INLINE inline __attribute__((always_inline))
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define INLINE inline
#endif


/********
 * SIMD *
 ********/

typedef enum {
	NONE,
	SSE2,
	AVX2
} simd_e;

simd_e simd_type = NONE;

#include <emmintrin.h>   // SSE2
#include <immintrin.h> // AVX + AVX2

int32_t avx2_available() {
	return __builtin_cpu_supports("avx2");
}

static inline __m128i mm128_select_epi16(__m128i mask, __m128i a, __m128i b) {
	return _mm_or_si128(_mm_and_si128(mask, a),
											 _mm_andnot_si128(mask, b));
}


/**********************
 * Indexing the graph *
 **********************/

#define arc_key(x) ((x).a)
KRADIX_SORT_INIT(gwf_arc, gwf_arc_t, arc_key, 8)

// index the graph such that we can quickly access the neighbors of a vertex
void gwf_ed_index_arc_core(uint64_t *idx, uint32_t n_vtx, uint32_t n_arc, gwf_arc_t *arc)
{
	uint32_t i, st;
	radix_sort_gwf_arc(arc, arc + n_arc);
	for (st = 0, i = 1; i <= n_arc; ++i) {
		if (i == n_arc || arc[i].a>>32 != arc[st].a>>32) {
			uint32_t v = arc[st].a>>32;
			assert(v < n_vtx);
			idx[v] = (uint64_t)st << 32 | (i - st);
			st = i;
		}
	}
}

void gwf_ed_index(void *km, gwf_graph_t *g)
{
	KMALLOC(km, g->aux, g->n_vtx);
	gwf_ed_index_arc_core(g->aux, g->n_vtx, g->n_arc, g->arc);
}

// free the index
void gwf_cleanup(void *km, gwf_graph_t *g)
{
	kfree(km, g->aux);
	g->aux = 0;
}

/**************************************
 * Graph WaveFront with edit distance *
 **************************************/

#include "khashl.h" // make it compatible with kalloc
#include "kdq.h"
#include "kvec.h"

#define GWF_DIAG_SHIFT 0x40000000

static inline uint64_t gwf_gen_vd(uint32_t v, int32_t d)
{
	return (uint64_t)v<<32 | (GWF_DIAG_SHIFT + d);
}

/*
 * Diagonal interval
 */
typedef struct {
	uint64_t vd0, vd1;
} gwf_intv_t;

typedef kvec_t(gwf_intv_t) gwf_intv_v;

#define intvd_key(x) ((x).vd0)
KRADIX_SORT_INIT(gwf_intv, gwf_intv_t, intvd_key, 8)

static int gwf_intv_is_sorted(int32_t n_a, const gwf_intv_t *a)
{
	int32_t i;
	for (i = 1; i < n_a; ++i)
		if (a[i-1].vd0 > a[i].vd0) break;
	return (i == n_a);
}

void gwf_ed_print_intv(size_t n, gwf_intv_t *a) // for debugging only
{
	size_t i;
	for (i = 0; i < n; ++i)
		printf("Z\t%d\t%d\t%d\n", (int32_t)(a[i].vd0>>32), (int32_t)a[i].vd0 - GWF_DIAG_SHIFT, (int32_t)a[i].vd1 - GWF_DIAG_SHIFT);
}

// merge overlapping intervals; input must be sorted
static size_t gwf_intv_merge_adj(size_t n, gwf_intv_t *a)
{
	size_t i, k;
	uint64_t st, en;
	if (n == 0) return 0;
	st = a[0].vd0, en = a[0].vd1;
	for (i = 1, k = 0; i < n; ++i) {
		if (a[i].vd0 > en) {
			a[k].vd0 = st, a[k++].vd1 = en;
			st = a[i].vd0, en = a[i].vd1;
		} else en = en > a[i].vd1? en : a[i].vd1;
	}
	a[k].vd0 = st, a[k++].vd1 = en;
	return k;
}

// merge two sorted interval lists
static size_t gwf_intv_merge2(gwf_intv_t *a, size_t n_b, const gwf_intv_t *b, size_t n_c, const gwf_intv_t *c)
{
	size_t i = 0, j = 0, k = 0;
	while (i < n_b && j < n_c) {
		if (b[i].vd0 <= c[j].vd0)
			a[k++] = b[i++];
		else a[k++] = c[j++];
	}
	while (i < n_b) a[k++] = b[i++];
	while (j < n_c) a[k++] = c[j++];
	return gwf_intv_merge_adj(k, a);
}

/*
 * Diagonal
 */
typedef struct { // a diagonal
	uint64_t vd; // higher 32 bits: vertex ID; lower 32 bits: diagonal+0x4000000
	int32_t k;
	uint32_t xo; // higher 31 bits: anti diagonal; lower 1 bit: out-of-order or not
	int32_t t;
} gwf_diag_t;

typedef kvec_t(gwf_diag_t) gwf_diag_v;

#define ed_key(x) ((x).vd)
KRADIX_SORT_INIT(gwf_ed, gwf_diag_t, ed_key, 8)

KDQ_INIT(gwf_diag_t)

void gwf_ed_print_diag(size_t n, gwf_diag_t *a) // for debugging only
{
	size_t i;
	for (i = 0; i < n; ++i) {
		int32_t d = (int32_t)a[i].vd - GWF_DIAG_SHIFT;
		printf("Z\t%d\t%d\t%d\t%d\t%d\n", (int32_t)(a[i].vd>>32), d, a[i].k, d + a[i].k, a[i].xo>>1);
	}
}

// push (v,d,k) to the end of the queue
static inline void gwf_diag_push(void *km, gwf_diag_v *a, uint32_t v, int32_t d, int32_t k, uint32_t x, uint32_t ooo, int32_t t)
{
	gwf_diag_t *p;
	kv_pushp(gwf_diag_t, km, *a, &p);
	p->vd = gwf_gen_vd(v, d), p->k = k, p->xo = x<<1|ooo, p->t = t;
}

// determine the wavefront on diagonal (v,d)
static inline int32_t gwf_diag_update(gwf_diag_t *p, uint32_t v, int32_t d, int32_t k, uint32_t x, uint32_t ooo, int32_t t)
{
	uint64_t vd = gwf_gen_vd(v, d);
	if (p->vd == vd) {
		p->xo = p->k > k? p->xo : x<<1|ooo;
		p->t  = p->k > k? p->t : t;
		p->k  = p->k > k? p->k : k;
		return 0;
	}
	return 1;
}

static int gwf_diag_is_sorted(int32_t n_a, const gwf_diag_t *a)
{
	int32_t i;
	for (i = 1; i < n_a; ++i)
		if (a[i-1].vd > a[i].vd) break;
	return (i == n_a);
}

// sort a[]. This uses the gwf_diag_t::ooo field to speed up sorting.
static void gwf_diag_sort(int32_t n_a, gwf_diag_t *a, void *km, gwf_diag_v *ooo)
{
	int32_t i, j, k, n_b, n_c;
	gwf_diag_t *b, *c;

	kv_resize(gwf_diag_t, km, *ooo, n_a);
	for (i = 0, n_c = 0; i < n_a; ++i)
		if (a[i].xo&1) ++n_c;
	n_b = n_a - n_c;
	b = ooo->a, c = b + n_b;
	for (i = j = k = 0; i < n_a; ++i) {
		if (a[i].xo&1) c[k++] = a[i];
		else b[j++] = a[i];
	}
	radix_sort_gwf_ed(c, c + n_c);
	for (k = 0; k < n_c; ++k) c[k].xo &= 0xfffffffeU;

	i = j = k = 0;
	while (i < n_b && j < n_c) {
		if (b[i].vd <= c[j].vd)
			a[k++] = b[i++];
		else a[k++] = c[j++];
	}
	while (i < n_b) a[k++] = b[i++];
	while (j < n_c) a[k++] = c[j++];
}

// remove diagonals not on the wavefront
static int32_t gwf_diag_dedup(int32_t n_a, gwf_diag_t *a, void *km, gwf_diag_v *ooo)
{
	int32_t i, n, st;
	if (!gwf_diag_is_sorted(n_a, a))
		gwf_diag_sort(n_a, a, km, ooo);
	for (i = 1, st = 0, n = 0; i <= n_a; ++i) {
		if (i == n_a || a[i].vd != a[st].vd) {
			int32_t j, max_j = st;
			if (st + 1 < i)
				for (j = st + 1; j < i; ++j) // choose the far end (i.e. the wavefront)
					if (a[max_j].k < a[j].k) max_j = j;
			a[n++] = a[max_j];
			st = i;
		}
	}
	return n;
}

// use forbidden bands to remove diagonals not on the wavefront
static int32_t gwf_mixed_dedup(int32_t n_a, gwf_diag_t *a, int32_t n_b, gwf_intv_t *b)
{
	int32_t i = 0, j = 0, k = 0;
	while (i < n_a && j < n_b) {
		if (a[i].vd >= b[j].vd0 && a[i].vd < b[j].vd1) ++i;
		else if (a[i].vd >= b[j].vd1) ++j;
		else a[k++] = a[i++];
	}
	while (i < n_a) a[k++] = a[i++];
	return k;
}

/*
 * Traceback stack
 */
KHASHL_MAP_INIT(KH_LOCAL, gwf_map64_t, gwf_map64, uint64_t, int32_t, kh_hash_uint64, kh_eq_generic)

typedef struct {
	int32_t v;
	int32_t pre;
} gwf_trace_t;

typedef kvec_t(gwf_trace_t) gwf_trace_v;

static int32_t gwf_trace_push(void *km, gwf_trace_v *a, int32_t v, int32_t pre, gwf_map64_t *h)
{
	uint64_t key = (uint64_t)v << 32 | (uint32_t)pre;
	khint_t k;
	int absent;
	k = gwf_map64_put(h, key, &absent);
	if (absent) {
		gwf_trace_t *p;
		kv_pushp(gwf_trace_t, km, *a, &p);
		p->v = v, p->pre = pre;
		kh_val(h, k) = a->n - 1;
		return a->n - 1;
	}
	return kh_val(h, k);
}

/*
 * Core GWFA routine
 */
KHASHL_INIT(KH_LOCAL, gwf_set64_t, gwf_set64, uint64_t, kh_hash_dummy, kh_eq_generic)

typedef struct {
	void *km;
	gwf_set64_t *ha; // hash table for adjacency
	gwf_map64_t *ht; // hash table for traceback
	gwf_intv_v intv;
	gwf_intv_v tmp, swap;
	gwf_diag_v ooo;
	gwf_trace_v t;
	int8_t** tbm // traceback matrices
} gwf_edbuf_t;


// function to initialize and allocate memory for traceback matrices
static inline void gwf_init_trace_mat(gwf_edbuf_t* buf, gwf_graph_t* g, int32_t ql)
{
	KCALLOC(buf->km, buf->tbm, g->n_vtx);
	// iterate over all nodes to prepare tb matrix for this node
	for (int i = 0; i < g->n_vtx; ++i) {
		KCALLOC(buf->km, buf->tbm[i], (g->len[i]+2) * (ql+2) * sizeof(int32_t)); // allocate memory for this node
	}
}

// function to free tb matrices
static inline void gwf_delete_trace_mat(gwf_edbuf_t* buf, gwf_graph_t* g)
{
	for (int i = 0; i < g->n_vtx; ++i) {
		kfree(buf->km, buf->tbm[i]);
	}
	kfree(buf->km, buf->tbm);
}

static inline void gwf_print_trace_mat(FILE* file, gwf_edbuf_t* buf, gwf_graph_t* g, int32_t ql, const char* q)
{
	for (int i = 0; i < g->n_vtx; ++i) {
		fprintf(file, "node: %i\tnl: %i\n", i, g->len[i]);
		fprintf(file, "\t\t");
		for (int j = 0; j < g->len[i]; ++j) {
			fprintf(file, "%c\t", g->seq[i][j]);
		}
		fprintf(file, "\n");
		for (int j = 0; j <= ql+1; ++j) {
			if (j == 0 || j == ql+1) {
				fprintf(file, "\t");
			} else {
				fprintf(file, "%c\t", q[j-1]);
			}
			for (int k = 0; k <= g->len[i]+1; ++k) {
				fprintf(file, "%i\t", buf->tbm[i][j*(g->len[i]+2) + k]);
			} 
			fprintf(file, "\n");
		}
		fprintf(file, "\n");
	}
}

static inline void gwf_walk_trace_mat(gwf_edbuf_t* buf, gwf_path_t* path, gwf_graph_t* g, int32_t ql) {
	
	// iterate over path to get minimal string length
	int32_t len = 0;
	for (int i = 0; i < path->nv; ++i) {
		len += g->len[path->v[i]];
	}
	// fprintf(stderr, "len: %i\n", len);
	// fprintf(stderr, "path len: %i\n", path->nv);
	char* cigar = (char*)calloc(len, sizeof(char));
	int32_t char_index = 0;


	int32_t end_off = path->end_off;
	int32_t end_v = path->end_v;
	int32_t i_q = ql - 1, i_n = 0; // index of position in query and node
	int32_t pos, score;
	for (int i = path->nv; i > 0; --i) {
		int32_t v = path->v[i-1];
		i_n = g->len[v] - 1; // set node index to the rightmost position in the node
		if (v == end_v) { // check wether we are in the first traversed node
			i_n = end_off;
		}

		// buf->tbm[v][i_q * (g->len[v]+2) + i_n]
		pos = buf->tbm[v][(i_q+1) * (g->len[v]+2) + i_n + 1];
		while (1) { // travers the current node/matrix
			// fprintf(stderr, "entry: %i\tv: %i\ti_q: %i\ti_n: %i\n", pos, v, i_q, i_n);
			switch(pos) {
				case 1:
					--i_q;
					--i_n;
					cigar[char_index] = 'M';
					++char_index;
					break;
				case 2:
					--i_n;
					++score;
					cigar[char_index] = 'D';
					++char_index;
					break;
				case 3:
					--i_q;
					++score;
					cigar[char_index] = 'I';
					++char_index;
					break;
				case 4:
					--i_q;
					--i_n;
					++score;
					cigar[char_index] = 'X';
					++char_index;
					break;
				default:
					fprintf(stderr, "[gwfa]error: unkonwn tbm entry!\n");
					fprintf(stderr, "entry: %i\tv: %i\ti_q: %i\ti_n: %i\n", pos, v, i_q, i_n);
					exit(1);
			}

			if (char_index == len-1) {
				len = len * 2;
				cigar = realloc(cigar, len);
			}

			pos = buf->tbm[v][(i_q+1) * (g->len[v]+2) + i_n + 1];
			if ((i_q == -1) && (pos != 2)) { // if we reached the start of the query then we break
				break;
			} else if (i_n == -1 && (pos != 3)) { // if we reached the start of the node and the end we break
				break;
			}
		}

		if ((i_q == -1) && (pos != 2)) { // if we reached the start of the query then we break
			break;
		} else if (i_n == -1 && (pos != 3)) { // if we reached the start of the node and the end we break
			break;
		}
	}


	path->cigar = (char*)calloc(char_index + 2, sizeof(char));
	for (int j = 0, i = char_index - 1; i >= 0; --i, ++j) {
		path->cigar[j] = cigar[i];
	}

	path->cigar[char_index] = '\0';
	free(cigar);
}


// remove diagonals not on the wavefront
static int32_t gwf_dedup(gwf_edbuf_t *buf, int32_t n_a, gwf_diag_t *a)
{
	if (buf->intv.n + buf->tmp.n > 0) {
		if (!gwf_intv_is_sorted(buf->tmp.n, buf->tmp.a))
			radix_sort_gwf_intv(buf->tmp.a, buf->tmp.a + buf->tmp.n);
		kv_copy(gwf_intv_t, buf->km, buf->swap, buf->intv);
		kv_resize(gwf_intv_t, buf->km, buf->intv, buf->intv.n + buf->tmp.n);
		buf->intv.n = gwf_intv_merge2(buf->intv.a, buf->swap.n, buf->swap.a, buf->tmp.n, buf->tmp.a);
	}
	n_a = gwf_diag_dedup(n_a, a, buf->km, &buf->ooo);
	if (buf->intv.n > 0)
		n_a = gwf_mixed_dedup(n_a, a, buf->intv.n, buf->intv.a);
	return n_a;
}

// remove diagonals that lag far behind the furthest wavefront
static int32_t gwf_prune(int32_t n_a, gwf_diag_t *a, uint32_t max_lag)
{
	int32_t i, j;
	uint32_t max_x = 0;
	for (i = 0; i < n_a; ++i)
		max_x = max_x > a[i].xo>>1? max_x : a[i].xo>>1;
	if (max_x <= max_lag) return n_a; // no filtering
	for (i = j = 0; i < n_a; ++i)
		if ((a[i].xo>>1) + max_lag >= max_x)
			a[j++] = a[i];
	return j;
}

// reach the wavefront
static inline int32_t gwf_extend1(int32_t d, int32_t k, int32_t vl, const char *ts, int32_t ql, const char *qs)
{
	int32_t max_k = (ql - d < vl? ql - d : vl) - 1;
	const char *ts_ = ts + 1, *qs_ = qs + d + 1;
#if 0
	// int32_t i = k + d; while (k + 1 < g->len[v] && i + 1 < ql && g->seq[v][k+1] == q[i+1]) ++k, ++i;
	while (k < max_k && *(ts_ + k) == *(qs_ + k))
		++k;
#else
	uint64_t cmp = 0;
	while (k + 7 < max_k) {
		uint64_t x = *(uint64_t*)(ts_ + k); // warning: unaligned memory access
		uint64_t y = *(uint64_t*)(qs_ + k);
		cmp = x ^ y;
		if (cmp == 0) k += 8;
		else break;
	}
	if (cmp)
		k += __builtin_ctzl(cmp) >> 3; // on x86, this is done via the BSR instruction: https://www.felixcloutier.com/x86/bsr
	else if (k + 7 >= max_k)
		while (k < max_k && *(ts_ + k) == *(qs_ + k)) // use this for generic CPUs. It is slightly faster than the unoptimized version
			++k;
#endif
	return k;
}


// reach the wavefront with SSE2 enabled
static inline int32_t gwf_extend1_sse2(int32_t d, int32_t k, int32_t vl, const char *ts, int32_t ql, const char *qs)
{
	int32_t max_k = (ql - d < vl? ql - d : vl) - 1;
	const char *ts_ = ts + 1, *qs_ = qs + d + 1;

	while (k + 15 < max_k) {
		__m128i x = _mm_loadu_si128((const __m128i *)(ts_ + k));
		__m128i y = _mm_loadu_si128((const __m128i *)(qs_ + k));

		// byte-wise comparison
		__m128i eq = _mm_cmpeq_epi8(x, y);

		// create 16-bit mask: 1 = equal, 0 = mismatch
		uint32_t mask = (uint32_t)_mm_movemask_epi8(eq);

		if (mask == 0xFFFFu) {
			// no mismatch
			k += 16;
		} else {
			// we find a mismatch
			k += __builtin_ctz(~mask);
			return k;
		}
	}

	// scalar cleanup
	while (k < max_k && ts_[k] == qs_[k])
		++k;
	return k;
}


// reach the wavefront with AVX2
static inline int32_t gwf_extend1_avx2(int32_t d, int32_t k, int32_t vl, const char *ts, int32_t ql, const char *qs)
{
	int32_t max_k = (ql - d < vl? ql - d : vl) - 1;
	const char *ts_ = ts + 1, *qs_ = qs + d + 1;

	while (k + 31 < max_k) {
		__m256i x = _mm256_loadu_si256((const __m256i *)(ts_ + k));
		__m256i y = _mm256_loadu_si256((const __m256i *)(qs_ + k));

		// byte-wise comparison
		__m256i eq = _mm256_cmpeq_epi8(x, y);

		// create 16-bit mask: 1 = equal, 0 = mismatch
		uint32_t mask = (uint32_t)_mm256_movemask_epi8(eq);

		if (mask == 0xFFFFFFFFu) {
			// no mismatch
			k += 32;
		} else {
			// we find a mismatch
			k += __builtin_ctz(~mask);
			return k;
		}
	}

	// scalar cleanup
	while (k < max_k && ts_[k] == qs_[k])
		++k;
	return k;
}


// expand batch section accelerated with SSE2 SIMD
static inline int32_t gwf_expand_sse2(int32_t j, int32_t n, gwf_diag_t* a, gwf_diag_t* b)
{
	for (; j + 7 < n - 1; j += 8) {
			
		// load k values
		__m128i k0 = _mm_set_epi16((uint16_t)a[j+6].k,
															(uint16_t)a[j+5].k,
															(uint16_t)a[j+4].k,
															(uint16_t)a[j+3].k,
															(uint16_t)a[j+2].k,
															(uint16_t)a[j+1].k,
															(uint16_t)a[j].k,
															(uint16_t)a[j-1].k);

		__m128i k1 = _mm_set_epi16((uint16_t)a[j+7].k,
															(uint16_t)a[j+6].k,
															(uint16_t)a[j+5].k,
															(uint16_t)a[j+4].k,
															(uint16_t)a[j+3].k,
															(uint16_t)a[j+2].k,
															(uint16_t)a[j+1].k,
															(uint16_t)a[j].k);
		k1 = _mm_add_epi16(k1, _mm_set1_epi16(1)); // + 1, we move one

		__m128i k2 = _mm_set_epi16((uint16_t)a[j+8].k,
															(uint16_t)a[j+7].k,
															(uint16_t)a[j+6].k,
															(uint16_t)a[j+5].k,
															(uint16_t)a[j+4].k,
															(uint16_t)a[j+3].k,
															(uint16_t)a[j+2].k,
															(uint16_t)a[j+1].k);
		k2 = _mm_add_epi16(k2, _mm_set1_epi16(1)); // + 1, we move one

		// compare k0 and k1
		__m128i m01 = _mm_cmpgt_epi16(k0, k1);
		__m128i k01 = mm128_select_epi16(m01, k0, k1);

		// compare k01 and k2
		__m128i m = _mm_cmpgt_epi16(k01, k2);
		__m128i k = mm128_select_epi16(m, k01, k2);

		// build masks
		__m128i mask2 = _mm_andnot_si128(m, _mm_set1_epi16(-1));
		__m128i mask1 = _mm_andnot_si128(m01, m);
		__m128i mask0 = _mm_and_si128(m, m01);

		// load xo values
		__m128i xo0 = _mm_set_epi16((uint16_t)a[j+6].xo,
															(uint16_t)a[j+5].xo,
															(uint16_t)a[j+4].xo,
															(uint16_t)a[j+3].xo,
															(uint16_t)a[j+2].xo,
															(uint16_t)a[j+1].xo,
															(uint16_t)a[j].xo,
															(uint16_t)a[j-1].xo);
		xo0 = _mm_add_epi16(xo0, _mm_set1_epi16(2));

		__m128i xo1 = _mm_set_epi16((uint16_t)a[j+7].xo,
															(uint16_t)a[j+6].xo,
															(uint16_t)a[j+5].xo,
															(uint16_t)a[j+4].xo,
															(uint16_t)a[j+3].xo,
															(uint16_t)a[j+2].xo,
															(uint16_t)a[j+1].xo,
															(uint16_t)a[j].xo);
		xo1 = _mm_add_epi16(xo1, _mm_set1_epi16(4));

		__m128i xo2 = _mm_set_epi16((uint16_t)a[j+8].xo,
															(uint16_t)a[j+7].xo,
															(uint16_t)a[j+6].xo,
															(uint16_t)a[j+5].xo,
															(uint16_t)a[j+4].xo,
															(uint16_t)a[j+3].xo,
															(uint16_t)a[j+2].xo,
															(uint16_t)a[j+1].xo);
		xo2 = _mm_add_epi16(xo2, _mm_set1_epi16(2));

		// build xo results
		__m128i xo = 
			_mm_or_si128(
				_mm_or_si128(
					_mm_and_si128(mask0, xo0),
					_mm_and_si128(mask1, xo1)),
				_mm_and_si128(mask2, xo2));

		// load t values
		__m128i t0 = _mm_set_epi16((uint16_t)a[j+6].t,
															(uint16_t)a[j+5].t,
															(uint16_t)a[j+4].t,
															(uint16_t)a[j+3].t,
															(uint16_t)a[j+2].t,
															(uint16_t)a[j+1].t,
															(uint16_t)a[j].t,
															(uint16_t)a[j-1].t);

		__m128i t1 = _mm_set_epi16((uint16_t)a[j+7].t,
															(uint16_t)a[j+6].t,
															(uint16_t)a[j+5].t,
															(uint16_t)a[j+4].t,
															(uint16_t)a[j+3].t,
															(uint16_t)a[j+2].t,
															(uint16_t)a[j+1].t,
															(uint16_t)a[j].t);

		__m128i t2 = _mm_set_epi16((uint16_t)a[j+8].t,
															(uint16_t)a[j+7].t,
															(uint16_t)a[j+6].t,
															(uint16_t)a[j+5].t,
															(uint16_t)a[j+4].t,
															(uint16_t)a[j+3].t,
															(uint16_t)a[j+2].t,
															(uint16_t)a[j+1].t);

		// build t results
		__m128i t = 
			_mm_or_si128(
				_mm_or_si128(
					_mm_and_si128(mask0, t0),
					_mm_and_si128(mask1, t1)),
				_mm_and_si128(mask2, t2));

		// cast into scalar values
		uint16_t k_out[8], xo_out[8], t_out[8];
		_mm_storeu_si128((__m128i*)k_out, k);
		_mm_storeu_si128((__m128i*)xo_out, xo);
		_mm_storeu_si128((__m128i*)t_out, t);

		for (int lane = 0; lane < 8; ++lane) {
			b[j+1+lane].k = k_out[lane];
			b[j+1+lane].xo = xo_out[lane];
			b[j+1+lane].t = t_out[lane];
			b[j+1+lane].vd = a[j+lane].vd;
		}
	}

	return j;
}


// expand batch section accelerated with SSE2 SIMD
static inline int32_t gwf_expand_avx2(int32_t j, int32_t n, gwf_diag_t* a, gwf_diag_t* b)
{
	for (; j + 15 < n - 1; j += 16) {

		// load k values
		__m256i k0 = _mm256_set_epi16(
			(uint16_t)a[j+14].k, (uint16_t)a[j+13].k,
			(uint16_t)a[j+12].k, (uint16_t)a[j+11].k,
			(uint16_t)a[j+10].k, (uint16_t)a[j+9].k,
			(uint16_t)a[j+8].k,  (uint16_t)a[j+7].k,
			(uint16_t)a[j+6].k,  (uint16_t)a[j+5].k,
			(uint16_t)a[j+4].k,  (uint16_t)a[j+3].k,
			(uint16_t)a[j+2].k,  (uint16_t)a[j+1].k,
			(uint16_t)a[j].k,    (uint16_t)a[j-1].k
		);

		__m256i k1 = _mm256_set_epi16(
			(uint16_t)a[j+15].k, (uint16_t)a[j+14].k,
			(uint16_t)a[j+13].k, (uint16_t)a[j+12].k,
			(uint16_t)a[j+11].k, (uint16_t)a[j+10].k,
			(uint16_t)a[j+9].k,  (uint16_t)a[j+8].k,
			(uint16_t)a[j+7].k,  (uint16_t)a[j+6].k,
			(uint16_t)a[j+5].k,  (uint16_t)a[j+4].k,
			(uint16_t)a[j+3].k,  (uint16_t)a[j+2].k,
			(uint16_t)a[j+1].k,  (uint16_t)a[j].k
		);
		k1 = _mm256_add_epi16(k1, _mm256_set1_epi16(1)); // + 1, we move one

		__m256i k2 = _mm256_set_epi16(
			(uint16_t)a[j+16].k, (uint16_t)a[j+15].k,
			(uint16_t)a[j+14].k, (uint16_t)a[j+13].k,
			(uint16_t)a[j+12].k, (uint16_t)a[j+11].k,
			(uint16_t)a[j+10].k,  (uint16_t)a[j+9].k,
			(uint16_t)a[j+8].k,  (uint16_t)a[j+7].k,
			(uint16_t)a[j+6].k,  (uint16_t)a[j+5].k,
			(uint16_t)a[j+4].k,  (uint16_t)a[j+3].k,
			(uint16_t)a[j+2].k,  (uint16_t)a[j+1].k
		);
		k2 = _mm256_add_epi16(k2, _mm256_set1_epi16(1)); // + 1, we move one
		
		// compare k0 to k1
		__m256i m01 = _mm256_cmpgt_epi16(k0, k1);
		__m256i k01 = _mm256_blendv_epi8(k1, k0, m01);

		// compare k01 to k2
		__m256i m = _mm256_cmpgt_epi16(k01, k2);
		__m256i k = _mm256_blendv_epi8(k2, k01, m);

		// build masks
		__m256i mask2 = _mm256_andnot_si256(m, _mm256_set1_epi16(-1));
		__m256i mask1 = _mm256_andnot_si256(m01, m);
		__m256i mask0 = _mm256_and_si256(m, m01);

		// load xo values values
		__m256i xo0 = _mm256_set_epi16(
			(uint16_t)a[j+14].xo, (uint16_t)a[j+13].xo,
			(uint16_t)a[j+12].xo, (uint16_t)a[j+11].xo,
			(uint16_t)a[j+10].xo, (uint16_t)a[j+9].xo,
			(uint16_t)a[j+8].xo,  (uint16_t)a[j+7].xo,
			(uint16_t)a[j+6].xo,  (uint16_t)a[j+5].xo,
			(uint16_t)a[j+4].xo,  (uint16_t)a[j+3].xo,
			(uint16_t)a[j+2].xo,  (uint16_t)a[j+1].xo,
			(uint16_t)a[j].xo,    (uint16_t)a[j-1].xo
		);
		xo0 = _mm256_add_epi16(xo0, _mm256_set1_epi16(2));

		__m256i xo1 = _mm256_set_epi16(
			(uint16_t)a[j+15].xo, (uint16_t)a[j+14].xo,
			(uint16_t)a[j+13].xo, (uint16_t)a[j+12].xo,
			(uint16_t)a[j+11].xo, (uint16_t)a[j+10].xo,
			(uint16_t)a[j+9].xo,  (uint16_t)a[j+8].xo,
			(uint16_t)a[j+7].xo,  (uint16_t)a[j+6].xo,
			(uint16_t)a[j+5].xo,  (uint16_t)a[j+4].xo,
			(uint16_t)a[j+3].xo,  (uint16_t)a[j+2].xo,
			(uint16_t)a[j+1].xo,  (uint16_t)a[j].xo
		);
		xo1 = _mm256_add_epi16(xo1, _mm256_set1_epi16(4));

		__m256i xo2 = _mm256_set_epi16(
			(uint16_t)a[j+16].xo, (uint16_t)a[j+15].xo,
			(uint16_t)a[j+14].xo, (uint16_t)a[j+13].xo,
			(uint16_t)a[j+12].xo, (uint16_t)a[j+11].xo,
			(uint16_t)a[j+10].xo,  (uint16_t)a[j+9].xo,
			(uint16_t)a[j+8].xo,  (uint16_t)a[j+7].xo,
			(uint16_t)a[j+6].xo,  (uint16_t)a[j+5].xo,
			(uint16_t)a[j+4].xo,  (uint16_t)a[j+3].xo,
			(uint16_t)a[j+2].xo,  (uint16_t)a[j+1].xo
		);
		xo2 = _mm256_add_epi16(xo2, _mm256_set1_epi16(2));

		// build xo results
		__m256i xo = 
			_mm256_or_si256(
				_mm256_or_si256(
					_mm256_and_si256(mask0, xo0),
					_mm256_and_si256(mask1, xo1)),
				_mm256_and_si256(mask2, xo2));

		// load t values
		__m256i t0 = _mm256_set_epi16(
			(uint16_t)a[j+14].t, (uint16_t)a[j+13].t,
			(uint16_t)a[j+12].t, (uint16_t)a[j+11].t,
			(uint16_t)a[j+10].t, (uint16_t)a[j+9].t,
			(uint16_t)a[j+8].t,  (uint16_t)a[j+7].t,
			(uint16_t)a[j+6].t,  (uint16_t)a[j+5].t,
			(uint16_t)a[j+4].t,  (uint16_t)a[j+3].t,
			(uint16_t)a[j+2].t,  (uint16_t)a[j+1].t,
			(uint16_t)a[j].t,    (uint16_t)a[j-1].t
		);

		__m256i t1 = _mm256_set_epi16(
			(uint16_t)a[j+15].t, (uint16_t)a[j+14].t,
			(uint16_t)a[j+13].t, (uint16_t)a[j+12].t,
			(uint16_t)a[j+11].t, (uint16_t)a[j+10].t,
			(uint16_t)a[j+9].t,  (uint16_t)a[j+8].t,
			(uint16_t)a[j+7].t,  (uint16_t)a[j+6].t,
			(uint16_t)a[j+5].t,  (uint16_t)a[j+4].t,
			(uint16_t)a[j+3].t,  (uint16_t)a[j+2].t,
			(uint16_t)a[j+1].t,  (uint16_t)a[j].t
		);

		__m256i t2 = _mm256_set_epi16(
			(uint16_t)a[j+16].t, (uint16_t)a[j+15].t,
			(uint16_t)a[j+14].t, (uint16_t)a[j+13].t,
			(uint16_t)a[j+12].t, (uint16_t)a[j+11].t,
			(uint16_t)a[j+10].t,  (uint16_t)a[j+9].t,
			(uint16_t)a[j+8].t,  (uint16_t)a[j+7].t,
			(uint16_t)a[j+6].t,  (uint16_t)a[j+5].t,
			(uint16_t)a[j+4].t,  (uint16_t)a[j+3].t,
			(uint16_t)a[j+2].t,  (uint16_t)a[j+1].t
		);

		// build t results
		__m256i t = 
			_mm256_or_si256(
				_mm256_or_si256(
					_mm256_and_si256(mask0, t0),
					_mm256_and_si256(mask1, t1)),
				_mm256_and_si256(mask2, t2));

		// cast into scalar values
		uint16_t k_out[16], xo_out[16], t_out[16];
		_mm256_storeu_si256((__m128i*)k_out, k);
		_mm256_storeu_si256((__m128i*)xo_out, xo);
		_mm256_storeu_si256((__m128i*)t_out, t);

		for (int lane = 0; lane < 16; ++lane) {
			b[j+1+lane].k = k_out[lane];
			b[j+1+lane].xo = xo_out[lane];
			b[j+1+lane].t = t_out[lane];
			b[j+1+lane].vd = a[j+lane].vd;
		}
	}

	return j;
}


// This is essentially Landau-Vishkin for linear sequences. The function speeds up alignment to long vertices. Not really necessary.
static void gwf_ed_extend_batch(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t n, gwf_diag_t *a, gwf_diag_v *B,
								kdq_t(gwf_diag_t) *A, gwf_intv_v *tmp_intv, int32_t traceback, gwf_edbuf_t* buf)
{
	int32_t i, j, m;
	int32_t v = a->vd>>32;
	int32_t vl = g->len[v];
	const char *ts = g->seq[v];
	gwf_diag_t *b;

	// wfa_extend
	for (j = 0; j < n; ++j) {
		int32_t k;
		if (UNLIKELY(simd_type == SSE2)) {
			k = gwf_extend1_sse2((int32_t)a[j].vd - GWF_DIAG_SHIFT, a[j].k, vl, ts, ql, q);
		} else if (LIKELY(simd_type == AVX2)) {
			k = gwf_extend1_avx2((int32_t)a[j].vd - GWF_DIAG_SHIFT, a[j].k, vl, ts, ql, q);
		} else {
			k = gwf_extend1((int32_t)a[j].vd - GWF_DIAG_SHIFT, a[j].k, vl, ts, ql, q);
		}

		a[j].xo += (k - a[j].k) << 2;
		a[j].k = k;
	}

	// wfa_next
	kv_resize(gwf_diag_t, km, *B, B->n + n + 2);
	b = &B->a[B->n];
	b[0].vd = a[0].vd - 1;
	b[0].xo = a[0].xo + 2; // 2 == 1<<1
	b[0].k = a[0].k + 1;
	b[0].t = a[0].t;


	b[1].vd = a[0].vd;
	b[1].xo =  n == 1 || a[0].k > a[1].k? a[0].xo + 4 : a[1].xo + 2;
	b[1].t  =  n == 1 || a[0].k > a[1].k? a[0].t : a[1].t;
	b[1].k  = (n == 1 || a[0].k > a[1].k? a[0].k : a[1].k) + 1;

	// set loop base
	j = 1;

	if (simd_type == SSE2) {

		j = gwf_expand_sse2(j, n, a, b);

	} else if (simd_type == AVX2) {

		j = gwf_expand_avx2(j, n, a, b);
		j = gwf_expand_sse2(j, n, a, b); // try to solve remaining section with sse2 instructions
	
	}

	// scalar tail
	for (; j < n - 1; ++j) {
		uint32_t x = a[j-1].xo + 2;
		int32_t k = a[j-1].k, t = a[j-1].t;
		int8_t tb = 0;
		x = k > a[j].k + 1? x : a[j].xo + 4;
		t = k > a[j].k + 1? t : a[j].t;
		k = k > a[j].k + 1? k : a[j].k + 1;

		x = k > a[j+1].k + 1? x : a[j+1].xo + 2;
		t = k > a[j+1].k + 1? t : a[j+1].t;
		k = k > a[j+1].k + 1? k : a[j+1].k + 1;
		b[j+1].vd = a[j].vd, b[j+1].k = k, b[j+1].xo = x, b[j+1].t = t;
	}
	if (n >= 2) {
		b[n].vd = a[n-1].vd;
		b[n].xo = a[n-2].k > a[n-1].k + 1? a[n-2].xo + 2 : a[n-1].xo + 4;
		b[n].t  = a[n-2].k > a[n-1].k + 1? a[n-2].t : a[n-1].t;
		b[n].k  = a[n-2].k > a[n-1].k + 1? a[n-2].k : a[n-1].k + 1;
	}
	b[n+1].vd = a[n-1].vd + 1;
	b[n+1].xo = a[n-1].xo + 2;
	b[n+1].t  = a[n-1].t;
	b[n+1].k  = a[n-1].k;

	// drop out-of-bound cells
	for (j = 0; j < n; ++j) {
		gwf_diag_t *p = &a[j];
		if (p->k == vl - 1 || (int32_t)p->vd - GWF_DIAG_SHIFT + p->k == ql - 1)
			p->xo |= 1, *kdq_pushp(gwf_diag_t, A) = *p;
	}
	for (j = 0, m = 0; j < n + 2; ++j) {
		gwf_diag_t *p = &b[j];
		int32_t d = (int32_t)p->vd - GWF_DIAG_SHIFT;
		if (d + p->k < ql && p->k < vl) {
			b[m++] = *p;
		} else if (p->k == vl) {
			gwf_intv_t *q;
			kv_pushp(gwf_intv_t, km, *tmp_intv, &q);
			q->vd0 = gwf_gen_vd(v, d), q->vd1 = q->vd0 + 1;
		}
	}
	B->n += m;
}

// wfa_extend and wfa_next combined
static gwf_diag_t *gwf_ed_extend(gwf_edbuf_t *buf, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v1, uint32_t max_lag, int32_t traceback,
								 int32_t *end_v, int32_t *end_off, int32_t *end_tb, int32_t *n_a_, gwf_diag_t *a)
{
	int32_t i, x, n = *n_a_, do_dedup = 1;
	kdq_t(gwf_diag_t) *A;
	gwf_diag_v B = {0,0,0};
	gwf_diag_t *b;

	*end_v = *end_off = *end_tb = -1;
	buf->tmp.n = 0;
	gwf_set64_clear(buf->ha); // hash table $h to avoid visiting a vertex twice
	for (i = 0, x = 1; i < 32; ++i, x <<= 1)
		if (x >= n) break;
	if (i < 4) i = 4;
	A = kdq_init2(gwf_diag_t, buf->km, i); // $A is a queue
	kv_resize(gwf_diag_t, buf->km, B, n * 2);
#if 0 // unoptimized version without calling gwf_ed_extend_batch() at all. The final result will be the same.
	A->count = n;
	memcpy(A->a, a, n * sizeof(*a));
#else // optimized for long vertices.
	for (x = 0, i = 1; i <= n; ++i) {
		if (i == n || a[i].vd != a[i-1].vd + 1) {
			gwf_ed_extend_batch(buf->km, g, ql, q, i - x, &a[x], &B, A, &buf->tmp, traceback, buf);
			x = i;
		}
	}
	if (kdq_size(A) == 0) do_dedup = 0;
#endif
	kfree(buf->km, a); // $a is not used as it has been copied to $A

	while (kdq_size(A)) {
		gwf_diag_t t;
		uint32_t x0;
		int32_t ooo, v, d, k, i, vl;

		t = *kdq_shift(gwf_diag_t, A);
		ooo = t.xo&1, v = t.vd >> 32; // vertex
		d = (int32_t)t.vd - GWF_DIAG_SHIFT; // diagonal
		k = t.k; // wavefront position on the vertex
		vl = g->len[v]; // $vl is the vertex length

		if (UNLIKELY(simd_type == SSE2)) {
			k = gwf_extend1_sse2(d, k, vl, g->seq[v], ql, q);
		} else if (LIKELY(simd_type == AVX2)) {
			k = gwf_extend1_avx2(d, k, vl, g->seq[v], ql, q);
		} else {
			k = gwf_extend1(d, k, vl, g->seq[v], ql, q);
		}

		i = k + d; // query position
		x0 = (t.xo >> 1) + ((k - t.k) << 1); // current anti diagonal

		if (k + 1 < vl && i + 1 < ql) { // the most common case: the wavefront is in the middle
			int32_t push1 = 1, push2 = 1;
			if (B.n >= 2) push1 = gwf_diag_update(&B.a[B.n - 2], v, d-1, k+1, x0 + 1, ooo, t.t);
			if (B.n >= 1) push2 = gwf_diag_update(&B.a[B.n - 1], v, d,   k+1, x0 + 2, ooo, t.t);
			if (push1) gwf_diag_push(buf->km, &B, v, d-1, k+1, x0 + 1, 1, t.t);
			if (push2 || push1) gwf_diag_push(buf->km, &B, v, d,   k+1, x0 + 2, 1, t.t);
			gwf_diag_push(buf->km, &B, v, d+1, k, x0 + 1, ooo, t.t);
		} else if (i + 1 < ql) { // k + 1 == g->len[v]; reaching the end of the vertex but not the end of query
			int32_t ov = g->aux[v]>>32, nv = (int32_t)g->aux[v], j, n_ext = 0, tw = -1;
			gwf_intv_t *p;
			kv_pushp(gwf_intv_t, buf->km, buf->tmp, &p);
			p->vd0 = gwf_gen_vd(v, d), p->vd1 = p->vd0 + 1;
			if (traceback) tw = gwf_trace_push(buf->km, &buf->t, v, t.t, buf->ht);
			for (j = 0; j < nv; ++j) { // traverse $v's neighbors
				uint32_t w = (uint32_t)g->arc[ov + j].a; // $w is next to $v
				int32_t ol = g->arc[ov + j].o;
				int absent;
				gwf_set64_put(buf->ha, (uint64_t)w<<32 | (i + 1), &absent); // test if ($w,$i) has been visited
				if (q[i + 1] == g->seq[w][ol]) { // can be extended to the next vertex without a mismatch
					++n_ext;
					if (absent) {
						gwf_diag_t *p;
						p = kdq_pushp(gwf_diag_t, A);
						p->vd = gwf_gen_vd(w, i+1-ol), p->k = ol, p->xo = (x0+2)<<1 | 1, p->t = tw;
					}
				} else if (absent) {
					gwf_diag_push(buf->km, &B, w, i-ol,   ol, x0 + 1, 1, tw);
					gwf_diag_push(buf->km, &B, w, i+1-ol, ol, x0 + 2, 1, tw);
				}
			}
			if (nv == 0 || n_ext != nv) // add an insertion to the target; this *might* cause a duplicate in corner cases
				gwf_diag_push(buf->km, &B, v, d+1, k, x0 + 1, 1, t.t);
		} else if (v1 < 0 || (v == v1 && k + 1 == vl)) { // i + 1 == ql
			*end_v = v, *end_off = k, *end_tb = t.t, *n_a_ = 0;
			kdq_destroy(gwf_diag_t, A);
			kfree(buf->km, B.a);
			return 0;
		} else if (k + 1 < vl) { // i + 1 == ql; reaching the end of the query but not the end of the vertex
			gwf_diag_push(buf->km, &B, v, d-1, k+1, x0 + 1, ooo, t.t); // add an deletion; this *might* case a duplicate in corner cases
		} else if (v != v1) { // i + 1 == ql && k + 1 == g->len[v]; not reaching the last vertex $v1
			int32_t ov = g->aux[v]>>32, nv = (int32_t)g->aux[v], j, tw = -1;
			if (traceback) tw = gwf_trace_push(buf->km, &buf->t, v, t.t, buf->ht);
			for (j = 0; j < nv; ++j) {
				uint32_t w = (uint32_t)g->arc[ov + j].a;
				int32_t ol = g->arc[ov + j].o;
				gwf_diag_push(buf->km, &b, w, i-ol, ol, x0 + 1, 1, tw); // deleting the first base on the next vertex
			}
		} else assert(0); // should never come here
	}

	kdq_destroy(gwf_diag_t, A);
	*n_a_ = n = B.n, b = B.a;

	if (do_dedup) *n_a_ = n = gwf_dedup(buf, n, b);
	if (max_lag > 0) *n_a_ = n = gwf_prune(n, b, max_lag);
	return b;
}

static void gwf_traceback(gwf_edbuf_t *buf, int32_t end_v, int32_t end_tb, gwf_path_t *path)
{
	int32_t i = end_tb, n = 1;
	while (i >= 0 && buf->t.a[i].v >= 0)
		++n, i = buf->t.a[i].pre;
	KMALLOC(buf->km, path->v, n);
	i = end_tb, n = 0;
	path->v[n++] = end_v;
	while (i >= 0 && buf->t.a[i].v >= 0)
		path->v[n++] = buf->t.a[i].v, i = buf->t.a[i].pre;
	path->nv = n;
	for (i = 0; i < path->nv>>1; ++i)
		n = path->v[i], path->v[i] = path->v[path->nv - 1 - i], path->v[path->nv - 1 - i] = n;
}

int32_t gwf_ed(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v0, int32_t v1, uint32_t max_lag, int32_t traceback, gwf_path_t *path)
{
	int32_t s = 0, n_a = 1, end_tb;
	gwf_diag_t *a;
	gwf_edbuf_t buf;
	char* cigar;

	memset(&buf, 0, sizeof(buf));
	buf.km = km;
	buf.ha = gwf_set64_init2(km);
	buf.ht = gwf_map64_init2(km);
	kv_resize(gwf_trace_t, km, buf.t, g->n_vtx + 16);
	KCALLOC(km, a, 1);
	a[0].vd = gwf_gen_vd(v0, 0), a[0].k = -1, a[0].xo = 0; // the initial state

	if (traceback) a[0].t = gwf_trace_push(km, &buf.t, -1, -1, buf.ht);
	if (traceback == 2) gwf_init_trace_mat(&buf, g, ql);
	while (n_a > 0) {
		a = gwf_ed_extend(&buf, g, ql, q, v1, max_lag, traceback, &path->end_v, &path->end_off, &end_tb, &n_a, a);
		if (path->end_off >= 0 || n_a == 0) break;
		++s;
#ifdef GWF_DEBUG
		// printf("[%s] dist=%d, n=%d, n_intv=%ld, n_tb=%ld\n", __func__, s, n_a, buf.intv.n, buf.t.n);
#endif
	}
	if (traceback) gwf_traceback(&buf, path->end_v, end_tb, path);
	// if (traceback == 2) {
	// 	FILE* outputFile = fopen("./test_file.txt", "w");
	// 	gwf_print_trace_mat(outputFile, &buf, g, ql, q);
	// 	fclose(outputFile);
	// }
	if (traceback == 2) gwf_walk_trace_mat(&buf, path, g, ql);
	if (traceback == 2) gwf_delete_trace_mat(&buf, g);
	gwf_set64_destroy(buf.ha);
	gwf_map64_destroy(buf.ht);
	kfree(km, buf.intv.a); kfree(km, buf.tmp.a); kfree(km, buf.swap.a); kfree(km, buf.t.a);
	path->s = path->end_v >= 0? s : -1;
	return path->s; // end_v < 0 could happen if v0 can't reach v1
}


int32_t gwf_ed_infix(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v0, int32_t v1, uint32_t max_lag, int32_t traceback, gwf_path_t *path)
{
	int32_t s = 0, n_a = 0, end_tb;
	gwf_diag_t *a;
	gwf_edbuf_t buf;

	memset(&buf, 0, sizeof(buf));
	buf.km = km;
	buf.ha = gwf_set64_init2(km);
	buf.ht = gwf_map64_init2(km);
	kv_resize(gwf_trace_t, km, buf.t, g->n_vtx + 16);
	KCALLOC(km, a, 1);
	gwf_diag_v vec;
	kv_init(vec);

	int32_t base_trace = gwf_trace_push(km, &buf.t, -1, -1, buf.ht);

	for (int i = 0; i < g->n_vtx; ++i) {
		for (int j = g->len[i] - 1; j >= 0; --j) {
			gwf_diag_t diag;
			diag.vd = gwf_gen_vd(i, -j-1);
			diag.k = j;
			diag.xo = j & ~1;
			if (traceback) {
				diag.t = base_trace;
			}
			kv_push(gwf_diag_t, km, vec, diag);
			n_a++;
		}
		gwf_diag_t diag;
		diag.vd = gwf_gen_vd(i, 0);
		diag.k = -1;
		diag.xo = -1 & ~1;
		if (traceback) {
			diag.t = base_trace;
		}
		kv_push(gwf_diag_t, km, vec, diag);
		n_a++;
	}

	if (traceback == 2) gwf_init_trace_mat(&buf, g, ql);

	a = vec.a;



	// int32_t s = 0, n_a = 1, end_tb;
	// gwf_diag_t *a;
	// gwf_edbuf_t buf;

	// memset(&buf, 0, sizeof(buf));
	// buf.km = km;
	// buf.ha = gwf_set64_init2(km);
	// buf.ht = gwf_map64_init2(km);
	// kv_resize(gwf_trace_t, km, buf.t, g->n_vtx + 16);
	// KCALLOC(km, a, 1);

	// a[0].vd = gwf_gen_vd(v0, 0), a[0].k = -1, a[0].xo = 0; // the initial state
	// if (traceback) a[0].t = gwf_trace_push(km, &buf.t, -1, -1, buf.ht);


	while (n_a > 0) {
		a = gwf_ed_extend(&buf, g, ql, q, v1, max_lag, traceback, &path->end_v, &path->end_off, &end_tb, &n_a, a);
		if (path->end_off >= 0 || n_a == 0) break;
		++s;
#ifdef GWF_DEBUG
		// printf("[%s] dist=%d, n=%d, n_intv=%ld, n_tb=%ld\n", __func__, s, n_a, buf.intv.n, buf.t.n);
#endif
	}
	if (traceback) gwf_traceback(&buf, path->end_v, end_tb, path);
	// if (traceback == 2) {
	// 	FILE* outputFile = fopen("./test_file.txt", "w");
	// 	gwf_print_trace_mat(outputFile, &buf, g, ql, q);
	// 	fclose(outputFile);
	// }
	// fprintf(stderr, "end_v: %i\tend_off: %i\n", path->end_v, path->end_off);
	if (traceback == 2) gwf_walk_trace_mat(&buf, path, g, ql);
	if (traceback == 2) gwf_delete_trace_mat(&buf, g);
	gwf_set64_destroy(buf.ha);
	gwf_map64_destroy(buf.ht);
	kfree(km, buf.intv.a); kfree(km, buf.tmp.a); kfree(km, buf.swap.a); kfree(km, buf.t.a);
	path->s = path->end_v >= 0? s : -1;
	return path->s; // end_v < 0 could happen if v0 can't reach v1
}


// gwfa extended with SIMD
int32_t gwf_ed_simd(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v0, int32_t v1, uint32_t max_lag, int32_t traceback, gwf_path_t *path)
{
	if (avx2_available()) {
		simd_type = AVX2;
	} else {
		simd_type = SSE2;
	}

	int32_t s = 0, n_a = 1, end_tb;
	gwf_diag_t *a;
	gwf_edbuf_t buf;
	char* cigar;

	memset(&buf, 0, sizeof(buf));
	buf.km = km;
	buf.ha = gwf_set64_init2(km);
	buf.ht = gwf_map64_init2(km);
	kv_resize(gwf_trace_t, km, buf.t, g->n_vtx + 16);
	KCALLOC(km, a, 1);
	a[0].vd = gwf_gen_vd(v0, 0), a[0].k = -1, a[0].xo = 0; // the initial state

	if (traceback) a[0].t = gwf_trace_push(km, &buf.t, -1, -1, buf.ht);
	if (traceback == 2) gwf_init_trace_mat(&buf, g, ql);
	while (n_a > 0) {
		a = gwf_ed_extend(&buf, g, ql, q, v1, max_lag, traceback, &path->end_v, &path->end_off, &end_tb, &n_a, a);
		if (path->end_off >= 0 || n_a == 0) break;
		++s;
#ifdef GWF_DEBUG
		// printf("[%s] dist=%d, n=%d, n_intv=%ld, n_tb=%ld\n", __func__, s, n_a, buf.intv.n, buf.t.n);
#endif
	}
	if (traceback) gwf_traceback(&buf, path->end_v, end_tb, path);
	// if (traceback == 2) {
	// 	FILE* outputFile = fopen("./test_file.txt", "w");
	// 	gwf_print_trace_mat(outputFile, &buf, g, ql, q);
	// 	fclose(outputFile);
	// }
	if (traceback == 2) gwf_walk_trace_mat(&buf, path, g, ql);
	if (traceback == 2) gwf_delete_trace_mat(&buf, g);
	gwf_set64_destroy(buf.ha);
	gwf_map64_destroy(buf.ht);
	kfree(km, buf.intv.a); kfree(km, buf.tmp.a); kfree(km, buf.swap.a); kfree(km, buf.t.a);
	path->s = path->end_v >= 0? s : -1;
	return path->s; // end_v < 0 could happen if v0 can't reach v1
}


int32_t gwf_ed_infix_simd(void *km, const gwf_graph_t *g, int32_t ql, const char *q, int32_t v0, int32_t v1, uint32_t max_lag, int32_t traceback, gwf_path_t *path)
{
	if (avx2_available()) {
		simd_type = AVX2;
	} else {
		simd_type = SSE2;
	}
	int32_t s = 0, n_a = 0, end_tb;
	gwf_diag_t *a;
	gwf_edbuf_t buf;

	memset(&buf, 0, sizeof(buf));
	buf.km = km;
	buf.ha = gwf_set64_init2(km);
	buf.ht = gwf_map64_init2(km);
	kv_resize(gwf_trace_t, km, buf.t, g->n_vtx + 16);
	KCALLOC(km, a, 1);
	gwf_diag_v vec;
	kv_init(vec);

	int32_t base_trace = gwf_trace_push(km, &buf.t, -1, -1, buf.ht);

	for (int i = 0; i < g->n_vtx; ++i) {
		for (int j = g->len[i] - 1; j >= 0; --j) {
			gwf_diag_t diag;
			diag.vd = gwf_gen_vd(i, -j-1);
			diag.k = j;
			diag.xo = j & ~1;
			if (traceback) {
				diag.t = base_trace;
			}
			kv_push(gwf_diag_t, km, vec, diag);
			n_a++;
		}
		gwf_diag_t diag;
		diag.vd = gwf_gen_vd(i, 0);
		diag.k = -1;
		diag.xo = -1 & ~1;
		if (traceback) {
			diag.t = base_trace;
		}
		kv_push(gwf_diag_t, km, vec, diag);
		n_a++;
	}

	if (traceback == 2) gwf_init_trace_mat(&buf, g, ql);

	a = vec.a;



	// int32_t s = 0, n_a = 1, end_tb;
	// gwf_diag_t *a;
	// gwf_edbuf_t buf;

	// memset(&buf, 0, sizeof(buf));
	// buf.km = km;
	// buf.ha = gwf_set64_init2(km);
	// buf.ht = gwf_map64_init2(km);
	// kv_resize(gwf_trace_t, km, buf.t, g->n_vtx + 16);
	// KCALLOC(km, a, 1);

	// a[0].vd = gwf_gen_vd(v0, 0), a[0].k = -1, a[0].xo = 0; // the initial state
	// if (traceback) a[0].t = gwf_trace_push(km, &buf.t, -1, -1, buf.ht);


	while (n_a > 0) {
		a = gwf_ed_extend(&buf, g, ql, q, v1, max_lag, traceback, &path->end_v, &path->end_off, &end_tb, &n_a, a);
		if (path->end_off >= 0 || n_a == 0) break;
		++s;
#ifdef GWF_DEBUG
		// printf("[%s] dist=%d, n=%d, n_intv=%ld, n_tb=%ld\n", __func__, s, n_a, buf.intv.n, buf.t.n);
#endif
	}
	if (traceback) gwf_traceback(&buf, path->end_v, end_tb, path);
	// if (traceback == 2) {
	// 	FILE* outputFile = fopen("./test_file.txt", "w");
	// 	gwf_print_trace_mat(outputFile, &buf, g, ql, q);
	// 	fclose(outputFile);
	// }
	// fprintf(stderr, "end_v: %i\tend_off: %i\n", path->end_v, path->end_off);
	if (traceback == 2) gwf_walk_trace_mat(&buf, path, g, ql);
	if (traceback == 2) gwf_delete_trace_mat(&buf, g);
	gwf_set64_destroy(buf.ha);
	gwf_map64_destroy(buf.ht);
	kfree(km, buf.intv.a); kfree(km, buf.tmp.a); kfree(km, buf.swap.a); kfree(km, buf.t.a);
	path->s = path->end_v >= 0? s : -1;
	return path->s; // end_v < 0 could happen if v0 can't reach v1
}
import config
import code
import cupy as cp

p = config.pixel
d = config.d
c = config.c
d_min = config.d_min
c_min = config.c_min
d_delta = config.d_delta
c_delta = config.c_delta
d_num = config.d_num
c_num = config.c_num
ker_num = config.ker_num

total_diag = min(4 * c + 1, p)
conv_code = code.conv_code()
pool_code = code.pool_code('j')
mod_p = "& {}".format(p - 1) if p in [2, 4, 8, 16, 32, 64] else "% {}".format(p)
array_size = p if p % 2 == 1 else p + 1

constant_code = code.constant_code({ 
	"d" : d,
	"d_delta" : d_delta,
	"d_min" : d_min,
	"d_num" : d_num,
	"c" : c,
	"c_delta" : c_delta,
	"c_min" : c_min,
	"c_num" : c_num,
	"ker_num" : ker_num,
	"total_diag" : total_diag,
	"pool_code" : pool_code,
	"p" : p,
	"mod_p" : mod_p,
	"p_minus_2" : p - 2,
	"q" : p - 1,
	"array_size" : array_size,
})

integrate = cp.RawKernel(constant_code + conv_code + r'''
extern "C" __global__
void integrate(const float s[p][p][p][p], float t[ker_num][d_num][c_num][total_diag][total_diag], const float Dx[p][p], const float Dz[p][p])
{
	int x1 = (threadIdx.x + blockIdx.x + 2 * (p - c)) mod_p;
	int y1 = (threadIdx.y + blockIdx.y + 2 * (p - c)) mod_p;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;
	int i, j, k;
	float U, S, P;
	__shared__ float a[4][array_size][array_size], b[4][array_size][array_size];

	a[1][x2][y2] = s[x1][y1][x2][y2];
	b[1][x2][y2] = a[1][x2][y2];
	a[3][x2][y2] = Dx[x2][y2];
	b[3][x2][y2] = Dz[x2][y2];
	__syncthreads();
	conv2(a[1], a[0], b[1], b[0], x1, y1, x2, y2);
	conv1(a[3], a[2], b[3], b[2], x2, y2);
	
	#pragma unroll
	for (i = 0; i <= d; ++ i) {
		__syncthreads();
		P = sqrtf(a[2][x1][y1] * b[2][x2][y2]);
		S = a[0][x2][y2] / P;
		U = INV_NINE - acosf(fmaxf(fminf(S, ONE), NEG_ONE)) * INV_NINE_PI;
		a[1][x2][y2] = a[0][x2][y2] * U + sqrtf(ONE - fminf(S * S, ONE)) * INV_NINE_PI * P;
		b[1][x2][y2] = b[0][x2][y2] * U + a[1][x2][y2];					
		if ((d - i) % d_delta == 0 && i >= d_min)
			#pragma unroll
			for (j = c; j >= c_min; j -= c_delta) {
				a[0][x2][y2] = a[1][x2][y2] * (pool_code);
				b[0][x2][y2] = b[1][x2][y2] * (pool_code);
				__syncthreads();
				if (y2 == 0) {
					#pragma unroll
					for (k = 1; k < p; ++ k) {
						a[0][x2][0] += a[0][x2][k];
						b[0][x2][0] += b[0][x2][k];
					}
				}
				__syncthreads();
				if (x2 == 0 && y2 == 0) {
					#pragma unroll
					for (k = 1; k < p; ++ k) {
						a[0][0][0] += a[0][k][0];
						b[0][0][0] += b[0][k][0];
					}
					t[0][(d - i) / d_delta][(c - j) / c_delta][blockIdx.x][blockIdx.y] = b[0][0][0];
					if (ker_num > 1)
						t[1][(d - i) / d_delta][(c - j) / c_delta][blockIdx.x][blockIdx.y] = a[0][0][0];
				}
				__syncthreads();
			}
		if (i == d)
			break;
		a[3][x1][y1] = a[2][x1][y1] * INV_NINE;	
		b[3][x2][y2] = b[2][x2][y2] * INV_NINE;
		__syncthreads(); 
		conv2(a[1], a[0], b[1], b[0], x1, y1, x2, y2);
		conv1(a[3], a[2], b[3], b[2], x2, y2);
	}   
}''', 'integrate', options=('--use_fast_math',))


blocks = (total_diag, total_diag)
threads = (p, p)
T = cp.zeros((ker_num, d_num, c_num, total_diag, total_diag), dtype = cp.float32)
def xz(x, z, Dx, Dz):
	global T
	S = cp.matmul(x.T, z).reshape(p, p, p, p)
	integrate(blocks, threads, (S, T, Dx, Dz))
	return cp.sum(T, axis = (3, 4))

import config
import code
import cupy as cp

p = config.pixel + 2
d = config.d
c = config.c
d_min = config.d_min
c_min = config.c_min
d_delta = config.d_delta
c_delta = config.c_delta
d_num = config.d_num
c_num = config.c_num
bias = config.bias
ker_num = config.ker_num

total_diag = min(4 * c + 1, p)
conv_code = code.conv_code()
pool_code = code.pool_code(config.pool_type)
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
    "array_size" : array_size,
    "bias" : str(bias) + "f"
})


integrate = cp.RawKernel(constant_code + conv_code + r'''
extern "C" __global__
void integrate(const float s[p - 2][p - 2][p - 2][p - 2], float t[ker_num][d_num][c_num][total_diag][total_diag], const float Dx[p - 2][p - 2], const float Dz[p - 2][p - 2])
{
	int x1 = (threadIdx.x + blockIdx.x + 2 * (p - c)) mod_p;
	int y1 = (threadIdx.y + blockIdx.y + 2 * (p - c)) mod_p;
	int x2 = threadIdx.x;
	int y2 = threadIdx.y;
	int i, j, k;
	float U, S, P;
    
	__shared__ float a[4][array_size][array_size], b[4][array_size][array_size];

    a[1][x2][y2] = 0.0f;
    a[3][x2][y2] = 0.0f;
    b[1][x2][y2] = 0.0f;
    b[3][x2][y2] = 0.0f;
    if (x2 > 0 && x2 < p - 1 && y2 > 0 && y2 < p - 1) { 
        if (x1 > 0 && x1 < p - 1 && y1 > 0 && y1 < p - 1) {
            a[1][x2][y2] = s[x1 - 1][y1 - 1][x2 - 1][y2 - 1];
            a[3][x2][y2] = a[1][x2][y2];
        }
        b[1][x2][y2] = Dx[x2 - 1][y2 - 1];
        b[3][x2][y2] = Dz[x2 - 1][y2 - 1];
    }
	__syncthreads();

    if (x2 > 0 && x2 < p - 1 && y2 > 0 && y2 < p - 1) { 
        if (x1 > 0 && x1 < p - 1 && y1 > 0 && y1 < p - 1) {
            conv(a[1], a[0], x2, y2);
            conv(a[3], a[2], x2, y2)
        }
        conv(b[1], b[0], x2, y2);
        conv(b[3], b[2], x2, y2);
    }
    else
        return;
	
	#pragma unroll
	for (i = 0; i <= d; ++ i) {
		__syncthreads();
        a[0][x2][y2] = a[0][x2][y2] * INV_NINE + bias;
        a[2][x2][y2] = a[2][x2][y2] * INV_NINE + bias;
        b[0][x2][y2] = b[0][x2][y2] * INV_NINE + bias;
        b[2][x2][y2] = b[2][x2][y2] * INV_NINE + bias;
        __syncthreads();
        P = sqrtf(b[0][x1][y1] * b[2][x2][y2]);
		S = a[0][x2][y2] / P;
		U = ONE - acosf(fmaxf(fminf(S, ONE), NEG_ONE)) * INV_PI;
        a[1][x2][y2] = a[0][x2][y2] * U + sqrtf(ONE - fminf(S * S, ONE)) * INV_PI * P;
        a[3][x2][y2] = a[2][x2][y2] * U + a[1][x2][y2];
        if (x1 == 0 || x1 == p - 1 || y1 == 0 || y1 == p - 1)
            a[1][x2][y2] = a[3][x2][y2] = 0.0f;

        if ((d - i) % d_delta == 0 && i >= d_min)
            #pragma unroll
            for (j = c; j >= c_min; j -= c_delta) {
                a[0][x2][y2] = a[1][x2][y2] * (pool_code);
                a[2][x2][y2] = a[3][x2][y2] * (pool_code);
                __syncthreads();
                if (y2 == 1) {
                    #pragma unroll
                    for (k = 2; k < p - 1; ++ k) {
                        a[0][x2][1] += a[0][x2][k];
                        a[2][x2][1] += a[2][x2][k];
                    }
                }
                __syncthreads();
                if (x2 == 1 && y2 == 1) {
                    #pragma unroll
                    for (k = 2; k < p - 1; ++ k) {
                        a[0][1][1] += a[0][k][1];
                        a[2][1][1] += a[2][k][1];
                    }
                    t[0][(d - i) / d_delta][(c - j) / c_delta][blockIdx.x][blockIdx.y] = a[2][1][1];
                    if (ker_num > 1)
                        t[1][(d - i) / d_delta][(c - j) / c_delta][blockIdx.x][blockIdx.y] = a[0][1][1];
                }
                __syncthreads();
            }
        if (i == d)
            break;
            
        b[1][x2][y2] = b[0][x2][y2];
        b[3][x2][y2] = b[2][x2][y2];
        
        __syncthreads(); 
        if (x1 > 0 && x1 < p - 1 && y1 > 0 && y1 < p - 1) {
            conv(a[1], a[0], x2, y2);
            conv(a[3], a[2], x2, y2)
        }
        conv(b[1], b[0], x2, y2);
        conv(b[3], b[2], x2, y2);
	}   
}''', 'integrate', options=('--use_fast_math',))


blocks = (total_diag, total_diag)
threads = (p, p)
T = cp.zeros((ker_num, d_num, c_num, total_diag, total_diag), dtype = cp.float32)
def xz(x, z, Dx, Dz):
	global T
	S = cp.matmul(x.T, z).reshape(p - 2, p - 2, p - 2, p - 2) 
	integrate(blocks, threads, (S, T, Dx, Dz))
	return cp.sum(T, axis = (3, 4))

(.venv) cc@mig1:~/mig_pp$ python3 mig_pp/milp/measure_constants.py --mode net --node A     # mig1
Node A (rank 0); rendezvous 10.191.130.146:29980, NIC enp161s0f0. Waiting up to 600s for the other node...

[1/2] gloo payload one-way time (ping-pong / 2)...
         kind   mb        bytes   one_way_us     GB/s
       decode    1         8192         99.3     0.08
       decode    2        16384        115.4     0.14
       decode    4        32768        146.0     0.22
       decode    8        65536        225.4     0.29
       decode   16       131072        210.7     0.62
       decode   32       262144        365.3     0.72
      prefill    1       524288        570.5     0.92
      prefill    2      1048576       1163.1     0.90
      prefill    4      2097152       2382.2     0.88
      prefill    8      4194304       4048.3     1.04
      prefill   16      8388608       8101.2     1.04
      prefill   32     16777216      16418.3     1.02
      fit: one_way = 115.2 us + bytes / 1.03 GB/s

[2/2] next_tokens one-way across nodes...
      batch   8: 106.0 us
      batch  16: 113.5 us
      batch  32: 114.1 us

==============================================================
Cross-node constants (same on both nodes, up to noise):
  NET_FIXED_US = 115.2
  NET_BW       = 1.033e+09   # bytes/s
  TOK_NET_US   = 114.1   # one-way, worst of the batches above

Per hop: GPU->pinned (PCIE_BW_PINNED) + NET_FIXED_US + bytes/NET_BW
         + pinned->GPU. HIDDEN=4096 sized the payloads; set it for
         other models, or rescale: the fit is per byte.
==============================================================




3 files changed, 165 insertions(+), 19 deletions(-)
(.venv) cc@mig2:~/mig_pp$ python3 mig_pp/milp/measure_constants.py --mode net --node B     # mig2
Node B (rank 1); rendezvous 10.191.130.146:29980, NIC enp161s0f0. Waiting up to 600s for the other node...

[1/2] gloo payload one-way time (ping-pong / 2)...
         kind   mb        bytes   one_way_us     GB/s
       decode    1         8192         99.4     0.08
       decode    2        16384        115.2     0.14
       decode    4        32768        145.7     0.22
       decode    8        65536        225.0     0.29
       decode   16       131072        210.3     0.62
       decode   32       262144        364.8     0.72
      prefill    1       524288        569.3     0.92
      prefill    2      1048576       1160.7     0.90
      prefill    4      2097152       2378.5     0.88
      prefill    8      4194304       4035.0     1.04
      prefill   16      8388608       8066.8     1.04
      prefill   32     16777216      16347.4     1.03
      fit: one_way = 116.3 us + bytes / 1.04 GB/s

[2/2] next_tokens one-way across nodes...
      batch   8: 105.9 us
      batch  16: 113.3 us
      batch  32: 113.9 us

==============================================================
Cross-node constants (same on both nodes, up to noise):
  NET_FIXED_US = 116.3
  NET_BW       = 1.038e+09   # bytes/s
  TOK_NET_US   = 113.9   # one-way, worst of the batches above

Per hop: GPU->pinned (PCIE_BW_PINNED) + NET_FIXED_US + bytes/NET_BW
         + pinned->GPU. HIDDEN=4096 sized the payloads; set it for
         other models, or rescale: the fit is per byte.
==============================================================
(.venv) cc@mig2:~/mig_pp$ 






Command - CUDA_VISIBLE_DEVICES=MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866 python3 mig_pp/milp/measure_constants.py 2>&1 | tee slice_3g20.log   # mig1


Results -


NVIDIA A100-PCIE-40GB MIG 3g.20gb: 3g.20gb of A100-40GB (card peak 1555 GB/s; shares: GPC 3/7, mem 20/40); lmsys/vicuna-7b-v1.5: hidden 4096, vocab 32000                                                                        
                                                                                                                                                                                                                                 
eff_weight (decode GEMMs, M tokens):                                                                                                                                                                                             
  M=1      606 GB/s   vs card 0.390  vs GPC 0.910  vs mem 0.780                                                                                                                                                                  
  M=8      581 GB/s   vs card 0.374  vs GPC 0.872  vs mem 0.748                                                                                                                                                                  
  M=32     609 GB/s   vs card 0.392  vs GPC 0.914  vs mem 0.784                                                                                                                                                                  
  mean     599 GB/s   vs card 0.385  vs GPC 0.899  vs mem 0.770                                                                                                                                                                  
                                                                                                                                                                                                                                 
eff_act: weight time 676 us, launch term 784 us (40 kernels x 19.6 us)                                                                                                                                                           
   mb  ctx  kernel us  act GB/s   ILP form vs card / GPC / mem                                                                                                                                                                   
    1   64        798         -   (no time left after weights + launches)                                                                                                                                                        
    1  192        814         -   (no time left after weights + launches)                                                                                                                                                        
    1  320        823         -   (no time left after weights + launches)                                                                                                                                                        
    1  448        832         -   (no time left after weights + launches)                                                                                                                                                        
    1  575        846         -   (no time left after weights + launches)                                                                                                                                                        
    8   64        869         -   (no time left after weights + launches)                                                                                                                                                        
    8  192        945         -   (no time left after weights + launches)                                                                                                                                                        
    8  320       1030         -   (no time left after weights + launches)                                                                                                                                                        
    8  448       1056         -   (no time left after weights + launches)                                                                                                                                                        
    8  575       1123         -   (no time left after weights + launches)                                                                                                                                                        
   32   64       1070         -   (no time left after weights + launches)                                                                                                                                                        
   32  192       1381         -   (no time left after weights + launches)                                                                                                                                                        
   32  320       1686     779.0   vs card 0.501  vs GPC 1.169  vs mem 1.002                                                                                                                                                      
   32  448       1998     452.4   vs card 0.291  vs GPC 0.679  vs mem 0.582                                                                                                                                                      
   32  575       2240     397.8   vs card 0.256  vs GPC 0.597  vs mem 0.512                                                                                                                                                      
  time-equivalent 472.7 GB/s   vs card 0.304  vs GPC 0.709  vs mem 0.608                                                                                                                                                         
                                                                                                                                                                                                                                 
copies, one activation of mb 8 (GB/s)   decode 64 KiB   prefill 4096 KiB                                                                                                                                                         
  D2H (pcie_bw)            1.32          18.90                                                                                                                                                                                   
  H2D (recv_h2d_bw)        1.90          17.58                                                                                                                                                                                   
                                                                                                                                                                                                                                 
prefill, one layer, seq 64 (kernel time):                                                                                                                                                                                        
   mb        us  TFLOP/s   bytes rate vs card / GPC / mem                                                                                                                                                                        
    1       941     27.6   vs card 0.334  vs GPC 0.780  vs mem 0.668                                                                                                                                                             
    4      1569     66.2   vs card 0.304  vs GPC 0.710  vs mem 0.609                                                                                                                                                             
   16      5825     71.3   vs card 0.194  vs GPC 0.452  vs mem 0.388                                                                                                                                                             
   32     11062     75.1   vs card 0.181  vs GPC 0.422  vs mem 0.361                                                                                                                                                             
                                                                                                                                                                                                                                 
endpoint kernels per microbatch (us):                                                                                                                                                                                            
                  embed    clone     norm  lm_head   argmax                                                                                                                                                                      
  decode mb2        3.0      1.5     18.5    387.4     17.8                                                                                                                                                                      
  decode mb8        5.5      1.6     21.6    396.2     18.2
  decode mb32       2.1      1.9     26.6    384.5     17.6
  prefill mb8       7.6      7.8    100.2    396.4     17.6

# measured on NVIDIA A100-PCIE-40GB MIG 3g.20gb, vicuna-7b-v1.5
SLICE_CONSTANTS["3g.20gb"] = dict(
    weight_rate=5.99e+11,  # bytes/s; eff_weight vs_card=0.385, vs_GPC=0.899, vs_mem=0.770
    act_rate=4.73e+11,  # bytes/s; eff_act vs_card=0.304, vs_GPC=0.709, vs_mem=0.608
    pcie_bw=1.32e+09,
    recv_h2d_bw=1.9e+09,
    launch_us=19.6,
    kernels_per_layer=40,
)



Command - CUDA_VISIBLE_DEVICES=MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5 python3 mig_pp/milp/measure_constants.py 2>&1 | tee slice_2g10.log   # mig1



results -

NVIDIA A100-PCIE-40GB MIG 2g.10gb: 2g.10gb of A100-40GB (card peak 1555 GB/s; shares: GPC 2/7, mem 10/40); lmsys/vicuna-7b-v1.5: hidden 4096, vocab 32000
                                                                                                                
eff_weight (decode GEMMs, M tokens):                                                                            
  M=1      331 GB/s   vs card 0.213  vs GPC 0.746  vs mem 0.852                                                 
  M=8      327 GB/s   vs card 0.210  vs GPC 0.735  vs mem 0.840                                                 
  M=32     317 GB/s   vs card 0.204  vs GPC 0.714  vs mem 0.816                                                 
  mean     325 GB/s   vs card 0.209  vs GPC 0.732  vs mem 0.836                                                 
                                                                                                                
eff_act: weight time 1245 us, launch term 806 us (40 kernels x 20.2 us)                                         
   mb  ctx  kernel us  act GB/s   ILP form vs card / GPC / mem                                                  
    1   64       1334         -   (no time left after weights + launches)                                       
    1  192       1360         -   (no time left after weights + launches)  
    1  320       1370         -   (no time left after weights + launches)                                       
    1  448       1383         -   (no time left after weights + launches)                                       
    1  575       1406         -   (no time left after weights + launches)                                       
    8   64       1446         -   (no time left after weights + launches)  
    8  192       1591         -   (no time left after weights + launches)                                       
    8  320       1737         -   (no time left after weights + launches)                                       
    8  448       1892         -   (no time left after weights + launches)                                       
    8  575       2026         -   (no time left after weights + launches)                                       
   32   64       1854         -   (no time left after weights + launches)                                       
   32  192       2422     294.9   vs card 0.190  vs GPC 0.664  vs mem 0.759                                     
   32  320       2987     188.5   vs card 0.121  vs GPC 0.424  vs mem 0.485                                     
   32  448       3574     160.0   vs card 0.103  vs GPC 0.360  vs mem 0.412                                     
   32  575       4121     149.9   vs card 0.096  vs GPC 0.337  vs mem 0.386                                     
  time-equivalent 171.4 GB/s   vs card 0.110  vs GPC 0.386  vs mem 0.441                                        
                                                                                                                
copies, one activation of mb 8 (GB/s)   decode 64 KiB   prefill 4096 KiB                                        
  D2H (pcie_bw)            1.27          18.33                                                                  
  H2D (recv_h2d_bw)        1.57          18.67                                                                  
                                                                                                                
prefill, one layer, seq 64 (kernel time):                                                                       
   mb        us  TFLOP/s   bytes rate vs card / GPC / mem           
    1      1582     16.4   vs card 0.199  vs GPC 0.696  vs mem 0.796                                                                                                                                                             
    4      2821     36.8   vs card 0.169  vs GPC 0.593  vs mem 0.677                                            
   16      8995     46.2   vs card 0.126  vs GPC 0.439  vs mem 0.502                                            
   32     19509     42.6   vs card 0.102  vs GPC 0.359  vs mem 0.410                                            
                                                                                                                
endpoint kernels per microbatch (us):                                                                           
                  embed    clone     norm  lm_head   argmax                               
  decode mb2        3.1      1.4     18.0    755.5     18.6                                                     
  decode mb8        5.8      1.6     21.3    776.4     20.9    
  decode mb32       2.2      2.1     27.1    759.7     22.4                                                                                                                                                                      
  prefill mb8      14.6     12.8    212.4    777.3     19.2                                                     
                                                                                                                
# measured on NVIDIA A100-PCIE-40GB MIG 2g.10gb, vicuna-7b-v1.5                                                 
SLICE_CONSTANTS["2g.10gb"] = dict(                                                                                                                                                                                               
    weight_rate=3.25e+11,  # bytes/s; eff_weight vs_card=0.209, vs_GPC=0.732, vs_mem=0.836
    act_rate=1.71e+11,  # bytes/s; eff_act vs_card=0.110, vs_GPC=0.386, vs_mem=0.441
    pcie_bw=1.27e+09,    
    recv_h2d_bw=1.57e+09,                                                                                                                                                                                                        
    launch_us=20.2,      
    kernels_per_layer=40,                                                                                                                                                                                                        
)                                                                         



Command - CUDA_VISIBLE_DEVICES=MIG-952b3e2f-2d64-5456-9930-c90a85a45f62 python3 mig_pp/milp/measure_constants.py 2>&1 | tee slice_1g5.log    # mig2


result - 

NVIDIA A100-PCIE-40GB MIG 1g.5gb: 1g.5gb of A100-40GB (card peak 1555 GB/s; shares: GPC 1/7, mem 5/40); lmsys/vicuna-7b-v1.5: hidden 4096, vocab 32000

eff_weight (decode GEMMs, M tokens):
  M=1      179 GB/s   vs card 0.115  vs GPC 0.807  vs mem 0.922
  M=8      178 GB/s   vs card 0.114  vs GPC 0.800  vs mem 0.914
  M=32     171 GB/s   vs card 0.110  vs GPC 0.771  vs mem 0.881
  mean     176 GB/s   vs card 0.113  vs GPC 0.792  vs mem 0.906

eff_act: weight time 2300 us, launch term 806 us (40 kernels x 20.1 us)
   mb  ctx  kernel us  act GB/s   ILP form vs card / GPC / mem
    1   64       2395         -   (no time left after weights + launches)
    1  192       2436         -   (no time left after weights + launches)
    1  320       2468         -   (no time left after weights + launches)
    1  448       2511         -   (no time left after weights + launches)
    1  575       2561         -   (no time left after weights + launches)
    8   64       2673         -   (no time left after weights + launches)
    8  192       2959         -   (no time left after weights + launches)
    8  320       3253     299.5   vs card 0.193  vs GPC 1.348  vs mem 1.541
    8  448       3552     136.4   vs card 0.088  vs GPC 0.614  vs mem 0.702
    8  575       3817     109.0   vs card 0.070  vs GPC 0.490  vs mem 0.561
   32   64       3486     111.0   vs card 0.071  vs GPC 0.500  vs mem 0.571
   32  192       4592      73.6   vs card 0.047  vs GPC 0.331  vs mem 0.378
   32  320       5721      67.5   vs card 0.043  vs GPC 0.304  vs mem 0.347
   32  448       6875      64.6   vs card 0.042  vs GPC 0.291  vs mem 0.332
   32  575       7982      63.6   vs card 0.041  vs GPC 0.286  vs mem 0.327
  time-equivalent 73.7 GB/s   vs card 0.047  vs GPC 0.332  vs mem 0.379

copies, one activation of mb 8 (GB/s)   decode 64 KiB   prefill 4096 KiB
  D2H (pcie_bw)            1.28          19.61
  H2D (recv_h2d_bw)        1.90          18.50

prefill, one layer, seq 64 (kernel time):
   mb        us  TFLOP/s   bytes rate vs card / GPC / mem
    1      2950      8.8   vs card 0.107  vs GPC 0.746  vs mem 0.853
    4      5582     18.6   vs card 0.086  vs GPC 0.599  vs mem 0.684
   16     19620     21.2   vs card 0.058  vs GPC 0.403  vs mem 0.461
   32     43226     19.2   vs card 0.046  vs GPC 0.324  vs mem 0.370

endpoint kernels per microbatch (us):
                  embed    clone     norm  lm_head   argmax
  decode mb2        3.1      1.4     18.1   1433.5     17.1
  decode mb8        5.9      1.7     21.6   1455.4     19.6
  decode mb32       2.5      2.7     33.4   1592.9     27.6
  prefill mb8      51.1     53.0    485.8   1455.2     16.8

# measured on NVIDIA A100-PCIE-40GB MIG 1g.5gb, vicuna-7b-v1.5
SLICE_CONSTANTS["1g.5gb"] = dict(
    weight_rate=1.76e+11,  # bytes/s; eff_weight vs_card=0.113, vs_GPC=0.792, vs_mem=0.906
    act_rate=7.37e+10,  # bytes/s; eff_act vs_card=0.047, vs_GPC=0.332, vs_mem=0.379
    pcie_bw=1.28e+09,
    recv_h2d_bw=1.9e+09,
    launch_us=20.1,
    kernels_per_layer=40,
)



### Post qwen 14b change

command - CUDA_VISIBLE_DEVICES=MIG-952b3e2f-2d64-5456-9930-c90a85a45f62 python3 mig_pp/milp/measure_constants.py 2>&1 | tee slice_1g5_qwen14b.log


results-

NVIDIA A100-PCIE-40GB MIG 1g.5gb: 1g.5gb of A100-40GB (card peak 1555 GB/s; shares: GPC 1/7, mem 5/40); ~/models/Qwen2.5-14B: hidden 5120, vocab 152064

eff_weight (decode GEMMs, M tokens):
  M=1      181 GB/s   vs card 0.116  vs GPC 0.815  vs mem 0.932
  M=8      179 GB/s   vs card 0.115  vs GPC 0.807  vs mem 0.922
  M=32     163 GB/s   vs card 0.105  vs GPC 0.733  vs mem 0.838
  mean     174 GB/s   vs card 0.112  vs GPC 0.785  vs mem 0.897

eff_act: weight time 3156 us, launch term 813 us (40 kernels x 20.3 us)
   mb  ctx  kernel us  act GB/s   ILP form vs card / GPC / mem
    1   64       3194         -   (no time left after weights + launches)
    1  192       3205         -   (no time left after weights + launches)
    1  320       3215         -   (no time left after weights + launches)
    1  448       3222         -   (no time left after weights + launches)
    1  575       3233         -   (no time left after weights + launches)
    8   64       3423         -   (no time left after weights + launches)
    8  192       3485         -   (no time left after weights + launches)
    8  320       3571         -   (no time left after weights + launches)
    8  448       3655         -   (no time left after weights + launches)
    8  575       3728         -   (no time left after weights + launches)
   32   64       3772         -   (no time left after weights + launches)
   32  192       4056     404.8   vs card 0.260  vs GPC 1.822  vs mem 2.083
   32  320       4358     134.3   vs card 0.086  vs GPC 0.605  vs mem 0.691
   32  448       4643     102.4   vs card 0.066  vs GPC 0.461  vs mem 0.527
   32  575       4913      90.8   vs card 0.058  vs GPC 0.409  vs mem 0.467
  time-equivalent 115.7 GB/s   vs card 0.074  vs GPC 0.521  vs mem 0.595

copies, one activation of mb 8 (GB/s)   decode 80 KiB   prefill 5120 KiB
  D2H (pcie_bw)            1.48          20.74
  H2D (recv_h2d_bw)        2.34          19.54

prefill, one layer, seq 64 (kernel time):
   mb        us  TFLOP/s   bytes rate vs card / GPC / mem
    1      3838      9.2   vs card 0.099  vs GPC 0.690  vs mem 0.788
    4      7942     17.8   vs card 0.057  vs GPC 0.397  vs mem 0.454
   16     29022     19.5   vs card 0.025  vs GPC 0.178  vs mem 0.204
   32     59891     18.9   vs card 0.019  vs GPC 0.132  vs mem 0.150

endpoint kernels per microbatch (us):
                  embed    clone     norm  lm_head   argmax
  decode mb2        3.2      1.5     18.6   8446.4     23.2
  decode mb8        6.0      1.7     22.8   8580.9     31.5
  decode mb32       2.8      3.0     37.0   9609.7     76.9
  prefill mb8      64.3     65.2    608.5   8584.2     31.5

# measured on NVIDIA A100-PCIE-40GB MIG 1g.5gb, Qwen2.5-14B
SLICE_CONSTANTS["1g.5gb"] = dict(
    weight_rate=1.74e+11,  # bytes/s; eff_weight vs_card=0.112, vs_GPC=0.785, vs_mem=0.897
    act_rate=1.16e+11,  # bytes/s; eff_act vs_card=0.074, vs_GPC=0.521, vs_mem=0.595
    pcie_bw=1.48e+09,
    recv_h2d_bw=2.34e+09,
    launch_us=20.3,
    kernels_per_layer=40,
)


### contention

commands - CUDA_VISIBLE_DEVICES=MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866,MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5 python3 mig_pp/milp/measure_constants.py 2>&1 | tee contention_A.log


results-
(.venv) cc@mig1:~/mig_pp$ CUDA_VISIBLE_DEVICES=MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866,MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5 python3 mig_pp/milp/measure_constants.py 2>&1 | tee contention_A.log
2 slices; each copies 3 s alone, then all together (hidden 4096, mb 8)

decode activation, 64 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866       1.59      1.33   0.83       2.03      2.15   1.06
  MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5       1.60      1.33   0.83       2.03      2.15   1.06
  card total D2H: alone-sum 3.19, together 2.66 GB/s

prefill activation, 4096 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866      20.71      1.82   0.09      19.19     10.63   0.55
  MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5      20.75      1.82   0.09      19.21     10.63   0.55
  card total D2H: alone-sum 41.47, together 3.64 GB/s
(.venv) cc@mig1:~/mig_pp$ 



command - CUDA_VISIBLE_DEVICES=MIG-952b3e2f-2d64-5456-9930-c90a85a45f62,MIG-175e1cbc-85ba-5f44-ac48-deac1590e175 python3 mig_pp/milp/measure_constants.py 2>&1 | tee contention_B.log


results - 

2 slices; each copies 3 s alone, then all together (hidden 4096, mb 8)

decode activation, 64 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-952b3e2f-2d64-5456-9930-c90a85a45f62       1.52      1.39   0.91       2.03      2.14   1.05
  MIG-175e1cbc-85ba-5f44-ac48-deac1590e175       1.54      1.40   0.91       2.04      2.14   1.05
  card total D2H: alone-sum 3.06, together 2.79 GB/s

prefill activation, 4096 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-952b3e2f-2d64-5456-9930-c90a85a45f62      20.58      1.77   0.09      19.14     10.73   0.56
  MIG-175e1cbc-85ba-5f44-ac48-deac1590e175      20.67      1.75   0.08      19.26     10.73   0.56
  card total D2H: alone-sum 41.25, together 3.52 GB/s
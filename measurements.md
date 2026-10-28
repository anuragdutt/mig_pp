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



  ### post eff_act

  command - CUDA_VISIBLE_DEVICES=MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866 python3 mig_pp/milp/measure_constants.py 2>&1 | tee slice_3g20_v2.log


results-
NVIDIA A100-PCIE-40GB MIG 3g.20gb: 3g.20gb of A100-40GB (card peak 1555 GB/s; shares: GPC 3/7, mem 20/40); lmsys/vicuna-7b-v1.5: hidden 4096, vocab 32000

eff_weight (decode GEMMs, M tokens):
  M=1      654 GB/s   vs card 0.420  vs GPC 0.981  vs mem 0.841
  M=8      612 GB/s   vs card 0.393  vs GPC 0.918  vs mem 0.787
  M=32     644 GB/s   vs card 0.414  vs GPC 0.966  vs mem 0.828
  mean     636 GB/s   vs card 0.409  vs GPC 0.955  vs mem 0.819

eff_act: weight time 636 us, launch term 793 us (40 kernels x 19.8 us)
   mb  ctx  kernel us  ILP GB/s  no-launch GB/s   no-launch vs card / GPC / mem
    1   64        764         -            10.3   vs card 0.007  vs GPC 0.015  vs mem 0.013
    1  192        777         -            24.2   vs card 0.016  vs GPC 0.036  vs mem 0.031
    1  320        789         -            35.9   vs card 0.023  vs GPC 0.054  vs mem 0.046
    1  448        791         -            49.2   vs card 0.032  vs GPC 0.074  vs mem 0.063
    1  575        802         -            58.3   vs card 0.037  vs GPC 0.087  vs mem 0.075
    8   64        823         -            56.5   vs card 0.036  vs GPC 0.085  vs mem 0.073
    8  192        895         -           105.7   vs card 0.068  vs GPC 0.159  vs mem 0.136
    8  320        974         -           130.4   vs card 0.084  vs GPC 0.196  vs mem 0.168
    8  448       1055         -           145.1   vs card 0.093  vs GPC 0.218  vs mem 0.187
    8  575       1126         -           158.1   vs card 0.102  vs GPC 0.237  vs mem 0.203
   32   64       1070         -            97.2   vs card 0.063  vs GPC 0.146  vs mem 0.125
   32  192       1383         -           146.4   vs card 0.094  vs GPC 0.220  vs mem 0.188
   32  320       1686     685.8           168.0   vs card 0.108  vs GPC 0.252  vs mem 0.216
   32  448       1999     427.4           178.7   vs card 0.115  vs GPC 0.268  vs mem 0.230
   32  575       2240     382.3           193.3   vs card 0.124  vs GPC 0.290  vs mem 0.249
  time-equivalent ILP form  445.6 GB/s   vs card 0.287  vs GPC 0.669  vs mem 0.573
  time-equivalent no-launch 147.9 GB/s   vs card 0.095  vs GPC 0.222  vs mem 0.190

copies, one activation of mb 8 (GB/s)   decode 64 KiB   prefill 4096 KiB
  D2H (pcie_bw)            1.37          20.06
  H2D (recv_h2d_bw)        2.01          19.19

prefill, one layer, seq 64 (kernel time):
   mb        us  TFLOP/s   bytes rate vs card / GPC / mem
    1       940     27.6   vs card 0.335  vs GPC 0.781  vs mem 0.669
    4      1568     66.3   vs card 0.305  vs GPC 0.711  vs mem 0.609
   16      5824     71.4   vs card 0.194  vs GPC 0.453  vs mem 0.388
   32     11061     75.1   vs card 0.181  vs GPC 0.422  vs mem 0.361

endpoint kernels per microbatch (us):
                  embed    clone     norm  lm_head   argmax
  decode mb2        3.0      1.4     18.6    386.5     19.2
  decode mb8        5.6      1.6     21.5    396.0     19.0
  decode mb32       2.1      1.9     26.7    384.6     17.9
  prefill mb8       7.5      7.7     99.9    396.5     17.4

# measured on NVIDIA A100-PCIE-40GB MIG 3g.20gb, vicuna-7b-v1.5
SLICE_CONSTANTS["3g.20gb"] = dict(
    weight_rate=6.36e+11,  # bytes/s; eff_weight vs_card=0.409, vs_GPC=0.955, vs_mem=0.819
    act_rate=4.46e+11,  # bytes/s, ILP form; eff_act vs_card=0.287, vs_GPC=0.669, vs_mem=0.573
    act_rate_no_launch=1.48e+11,  # bytes/s; eff_act vs_card=0.095, vs_GPC=0.222, vs_mem=0.190
    pcie_bw=1.37e+09,
    recv_h2d_bw=2.01e+09,
    launch_us=19.8,
    kernels_per_layer=40,


command  - CUDA_VISIBLE_DEVICES=MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5 python3 mig_pp/milp/measure_constants.py 2>&1 | tee slice_2g10_v2.log

results - 
NVIDIA A100-PCIE-40GB MIG 2g.10gb: 2g.10gb of A100-40GB (card peak 1555 GB/s; shares: GPC 2/7, mem 10/40); lmsys/vicuna-7b-v1.5: hidden 4096, vocab 32000

eff_weight (decode GEMMs, M tokens):
  M=1      343 GB/s   vs card 0.221  vs GPC 0.773  vs mem 0.883
  M=8      340 GB/s   vs card 0.219  vs GPC 0.766  vs mem 0.876
  M=32     323 GB/s   vs card 0.208  vs GPC 0.727  vs mem 0.831
  mean     336 GB/s   vs card 0.216  vs GPC 0.755  vs mem 0.863

eff_act: weight time 1206 us, launch term 803 us (40 kernels x 20.1 us)
   mb  ctx  kernel us  ILP GB/s  no-launch GB/s   no-launch vs card / GPC / mem
    1   64       1338         -            10.0   vs card 0.006  vs GPC 0.022  vs mem 0.026
    1  192       1361         -            22.0   vs card 0.014  vs GPC 0.049  vs mem 0.057
    1  320       1365         -            34.6   vs card 0.022  vs GPC 0.078  vs mem 0.089
    1  448       1389         -            41.5   vs card 0.027  vs GPC 0.093  vs mem 0.107
    1  575       1404         -            49.1   vs card 0.032  vs GPC 0.110  vs mem 0.126
    8   64       1445         -            44.2   vs card 0.028  vs GPC 0.099  vs mem 0.114
    8  192       1589         -            71.4   vs card 0.046  vs GPC 0.161  vs mem 0.184
    8  320       1736         -            83.3   vs card 0.054  vs GPC 0.187  vs mem 0.214
    8  448       1896         -            88.3   vs card 0.057  vs GPC 0.199  vs mem 0.227
    8  575       2026    4571.9            94.6   vs card 0.061  vs GPC 0.213  vs mem 0.243
   32   64       1854         -            65.2   vs card 0.042  vs GPC 0.147  vs mem 0.168
   32  192       2428     260.8            89.5   vs card 0.058  vs GPC 0.201  vs mem 0.230
   32  320       2986     180.5            99.1   vs card 0.064  vs GPC 0.223  vs mem 0.255
   32  448       3575     155.5           102.8   vs card 0.066  vs GPC 0.231  vs mem 0.264
   32  575       4122     146.8           106.4   vs card 0.068  vs GPC 0.239  vs mem 0.274
  time-equivalent ILP form  180.1 GB/s   vs card 0.116  vs GPC 0.405  vs mem 0.463
  time-equivalent no-launch 90.9 GB/s   vs card 0.058  vs GPC 0.205  vs mem 0.234

copies, one activation of mb 8 (GB/s)   decode 64 KiB   prefill 4096 KiB
  D2H (pcie_bw)            1.34          20.01
  H2D (recv_h2d_bw)        1.97          19.14

prefill, one layer, seq 64 (kernel time):
   mb        us  TFLOP/s   bytes rate vs card / GPC / mem
    1      1579     16.4   vs card 0.199  vs GPC 0.697  vs mem 0.797
    4      2823     36.8   vs card 0.169  vs GPC 0.592  vs mem 0.677
   16      8990     46.2   vs card 0.126  vs GPC 0.440  vs mem 0.503
   32     19504     42.6   vs card 0.102  vs GPC 0.359  vs mem 0.410

endpoint kernels per microbatch (us):
                  embed    clone     norm  lm_head   argmax
  decode mb2        3.1      1.4     18.1    755.0     17.0
  decode mb8        5.8      1.6     21.4    777.2     18.9
  decode mb32       2.2      2.1     27.0    760.4     21.4
  prefill mb8      15.1     13.0    212.7    778.7     18.5

# measured on NVIDIA A100-PCIE-40GB MIG 2g.10gb, vicuna-7b-v1.5
SLICE_CONSTANTS["2g.10gb"] = dict(
    weight_rate=3.36e+11,  # bytes/s; eff_weight vs_card=0.216, vs_GPC=0.755, vs_mem=0.863
    act_rate=1.8e+11,  # bytes/s, ILP form; eff_act vs_card=0.116, vs_GPC=0.405, vs_mem=0.463
    act_rate_no_launch=9.09e+10,  # bytes/s; eff_act vs_card=0.058, vs_GPC=0.205, vs_mem=0.234
    pcie_bw=1.34e+09,
    recv_h2d_bw=1.97e+09,
    launch_us=20.1,
    kernels_per_layer=40,
)


command - CUDA_VISIBLE_DEVICES=MIG-952b3e2f-2d64-5456-9930-c90a85a45f62 python3 mig_pp/milp/measure_constants.py 2>&1 | tee slice_1g5_v2.log


results-
NVIDIA A100-PCIE-40GB MIG 1g.5gb: 1g.5gb of A100-40GB (card peak 1555 GB/s; shares: GPC 1/7, mem 5/40); ~/models/Qwen2.5-14B: hidden 5120, vocab 152064

eff_weight (decode GEMMs, M tokens):
  M=1      180 GB/s   vs card 0.116  vs GPC 0.812  vs mem 0.928
  M=8      179 GB/s   vs card 0.115  vs GPC 0.808  vs mem 0.923
  M=32     163 GB/s   vs card 0.105  vs GPC 0.734  vs mem 0.838
  mean     174 GB/s   vs card 0.112  vs GPC 0.785  vs mem 0.897

eff_act: weight time 3158 us, launch term 805 us (40 kernels x 20.1 us)
   mb  ctx  kernel us  act GB/s   ILP form vs card / GPC / mem
    1   64       3190         -   (no time left after weights + launches)
    1  192       3203         -   (no time left after weights + launches)
    1  320       3214         -   (no time left after weights + launches)
    1  448       3220         -   (no time left after weights + launches)
    1  575       3232         -   (no time left after weights + launches)
    8   64       3420         -   (no time left after weights + launches)
    8  192       3486         -   (no time left after weights + launches)
    8  320       3575         -   (no time left after weights + launches)
    8  448       3650         -   (no time left after weights + launches)
    8  575       3720         -   (no time left after weights + launches)
   32   64       3771         -   (no time left after weights + launches)
   32  192       4058     377.3   vs card 0.243  vs GPC 1.699  vs mem 1.941
   32  320       4352     134.7   vs card 0.087  vs GPC 0.606  vs mem 0.693
   32  448       4649     100.9   vs card 0.065  vs GPC 0.454  vs mem 0.519
   32  575       4915      90.1   vs card 0.058  vs GPC 0.405  vs mem 0.463
  time-equivalent 114.5 GB/s   vs card 0.074  vs GPC 0.515  vs mem 0.589

copies, one activation of mb 8 (GB/s)   decode 80 KiB   prefill 5120 KiB
  D2H (pcie_bw)            1.52          20.83
  H2D (recv_h2d_bw)        2.38          19.59

prefill, one layer, seq 64 (kernel time):
   mb        us  TFLOP/s   bytes rate vs card / GPC / mem
    1      3840      9.2   vs card 0.098  vs GPC 0.689  vs mem 0.788
    4      7943     17.8   vs card 0.057  vs GPC 0.397  vs mem 0.454
   16     29020     19.5   vs card 0.025  vs GPC 0.178  vs mem 0.204
   32     60018     18.8   vs card 0.019  vs GPC 0.131  vs mem 0.150

endpoint kernels per microbatch (us):
                  embed    clone     norm  lm_head   argmax
  decode mb2        3.2      1.5     18.6   8450.8     16.6
  decode mb8        6.0      1.7     22.8   8581.2     40.6
  decode mb32       2.8      3.0     37.1   9587.9    103.0
  prefill mb8      64.4     65.4    608.0   8588.5     26.9

# measured on NVIDIA A100-PCIE-40GB MIG 1g.5gb, Qwen2.5-14B
SLICE_CONSTANTS["1g.5gb"] = dict(
    weight_rate=1.74e+11,  # bytes/s; eff_weight vs_card=0.112, vs_GPC=0.785, vs_mem=0.897
    act_rate=1.14e+11,  # bytes/s; eff_act vs_card=0.074, vs_GPC=0.515, vs_mem=0.589
    pcie_bw=1.52e+09,
    recv_h2d_bw=2.38e+09,
    launch_us=20.1,
    kernels_per_layer=40,
)


### Post contention 2

Command - 
CUDA_VISIBLE_DEVICES=MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866,MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5 python3 mig_pp/milp/measure_constants.py 2>&1 | tee contention_A_v2.log
nvidia-smi topo -m | tee topo_A.log


results - 
(.venv) cc@mig1:~/mig_pp$ CUDA_VISIBLE_DEVICES=MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866,MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5 python3 mig_pp/milp/measure_constants.py 2>&1 | tee contention_A_v2.log
2 slices; each copies 3 s alone, then all together (hidden 4096, mb 8)

decode activation, 64 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866       1.64      1.30   0.80       2.08      2.13   1.02
  MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5       1.55      1.32   0.85       2.01      2.13   1.06
  card total D2H: alone-sum 3.19, together 2.62 GB/s

prefill activation, 4096 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-2c8eaa9b-ce72-56b6-8bd0-a54f2fc9d866      20.66      1.56   0.08      19.23      8.33   0.43
  MIG-67acbdb3-a3bc-5786-a4eb-ee64a17644f5      20.69      1.56   0.08      16.10      8.33   0.52
  card total D2H: alone-sum 41.35, together 3.12 GB/s
(.venv) cc@mig1:~/mig_pp$ nvidia-smi topo -m | tee topo_A.log
        GPU0    NIC0    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      SYS     64-127,192-255  1               N/A
NIC0    SYS      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks 

NIC Legend:

  NIC0: mlx5_0

(.venv) cc@mig1:~/mig_pp$ 


COmmands -
CUDA_VISIBLE_DEVICES=MIG-952b3e2f-2d64-5456-9930-c90a85a45f62,MIG-175e1cbc-85ba-5f44-ac48-deac1590e175 python3 mig_pp/milp/measure_constants.py 2>&1 | tee contention_B_v2.log
nvidia-smi topo -m | tee topo_B.log



results-

(.venv) cc@mig2:~/mig_pp$ CUDA_VISIBLE_DEVICES=MIG-952b3e2f-2d64-5456-9930-c90a85a45f62,MIG-175e1cbc-85ba-5f44-ac48-deac1590e175 python3 mig_pp/milp/measure_constants.py 2>&1 | tee contention_B_v2.log
2 slices; each copies 3 s alone, then all together (hidden 4096, mb 8)
nvidia-smi topo -m | tee topo_B.log
decode activation, 64 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-952b3e2f-2d64-5456-9930-c90a85a45f62       1.55      1.33   0.85       2.03      2.18   1.08
  MIG-175e1cbc-85ba-5f44-ac48-deac1590e175       1.59      1.34   0.85       2.04      2.18   1.07
  card total D2H: alone-sum 3.14, together 2.67 GB/s

prefill activation, 4096 KiB (GB/s):
  slice                                     D2H alone  together  ratio  H2D alone  together  ratio
  MIG-952b3e2f-2d64-5456-9930-c90a85a45f62      20.38      1.86   0.09      19.11     10.96   0.57
  MIG-175e1cbc-85ba-5f44-ac48-deac1590e175      20.25      1.81   0.09      19.15     10.88   0.57
  card total D2H: alone-sum 40.63, together 3.67 GB/s
(.venv) cc@mig2:~/mig_pp$ nvidia-smi topo -m | tee topo_B.log
        GPU0    NIC0    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      SYS     64-127,192-255  1               N/A
NIC0    SYS      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0

(.venv) cc@mig2:~/mig_pp$ 



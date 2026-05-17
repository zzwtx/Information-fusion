# TCVP 10-Iteration Parameter Search

Reference: quick_param_comparison_tcvp_fedadam_reg0001_20260515_155858

\n===== Iteration 01: tcvp_search_a02_reg0001 =====
Rationale: Lower alpha to reduce over-adaptation on already-strong datasets.
Summary: logs/quick_param_comparison_tcvp_search_a02_reg0001_20260515_170121/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a02-reg0001	+0.1373	+0.4390	+0.7751	+0.6119	+0.3746
dtd	TCVP-FedAdam-a02-reg0001	-0.7870	+3.9699	+0.1932	+2.2693	+0.7411
oxford_pets	TCVP-FedAdam-a02-reg0001	+0.0013	-0.7283	+0.0391	-0.3687	-0.1836
oxford_flowers	TCVP-FedAdam-a02-reg0001	-0.4545	+1.3865	+1.5107	+1.4471	+0.4963
eurosat	TCVP-FedAdam-a02-reg0001	+0.0511	+1.5333	+7.3564	+4.0563	+2.0537
Average	-	-0.2104	+1.3201	+1.9749	+1.6032	+0.6964
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 02: tcvp_search_a04_reg0001 =====
Rationale: Increase alpha moderately to test whether base/HM gains scale without alpha=0.5 local drop.
Summary: logs/quick_param_comparison_tcvp_search_a04_reg0001_20260515_175249/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a04-reg0001	+0.6170	+0.4390	+0.6441	+0.5446	+0.5808
dtd	TCVP-FedAdam-a04-reg0001	-0.0463	+3.3333	-0.0966	+1.7943	+0.8740
oxford_pets	TCVP-FedAdam-a04-reg0001	-0.0453	+0.8028	+0.1901	+0.5134	+0.2341
oxford_flowers	TCVP-FedAdam-a04-reg0001	-0.8597	+2.8015	+1.8653	+2.3428	+0.7415
eurosat	TCVP-FedAdam-a04-reg0001	-0.5444	+7.7309	+0.6718	+4.2910	+1.8733
Average	-	-0.1757	+3.0215	+0.6549	+1.8972	+0.8607
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 03: tcvp_search_a05_reg0001 =====
Rationale: Retest high alpha with mild residual regularization.
Summary: logs/quick_param_comparison_tcvp_search_a05_reg0001_20260515_184422/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a05-reg0001	-0.1597	+0.1937	+0.6441	+0.4252	+0.1327
dtd	TCVP-FedAdam-a05-reg0001	-2.9629	+6.4236	-2.3309	+2.2818	-0.3406
oxford_pets	TCVP-FedAdam-a05-reg0001	+0.3084	-0.5635	+0.2237	-0.1944	+0.0570
oxford_flowers	TCVP-FedAdam-a05-reg0001	-1.4885	+2.6306	+1.2270	+1.9404	+0.2260
eurosat	TCVP-FedAdam-a05-reg0001	-0.4577	+5.4357	+1.9026	+3.7732	+1.6578
Average	-	-0.9521	+2.8240	+0.3333	+1.6452	+0.3466
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 04: tcvp_search_a03_h16_reg0001 =====
Rationale: Shrink MetaNet to reduce overfitting and communication/update variance.
Summary: logs/quick_param_comparison_tcvp_search_a03_h16_reg0001_20260515_193602/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a03-h16-reg0001	+0.4364	+0.2260	+0.7205	+0.4801	+0.4582
dtd	TCVP-FedAdam-a03-h16-reg0001	-1.7129	+4.4097	+1.0628	+2.9106	+0.5989
oxford_pets	TCVP-FedAdam-a03-h16-reg0001	-0.1953	+0.2393	-0.1566	+0.0523	-0.0714
oxford_flowers	TCVP-FedAdam-a03-h16-reg0001	-0.1216	+2.4596	+2.2908	+2.3776	+1.1280
eurosat	TCVP-FedAdam-a03-h16-reg0001	+0.1089	+4.2595	-4.4000	-0.0288	+0.0401
Average	-	-0.2969	+2.3188	-0.0965	+1.1584	+0.4308
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 05: tcvp_search_a03_h64_reg0001 =====
Rationale: Increase MetaNet capacity to see whether underfitting limits alignment.
Summary: logs/quick_param_comparison_tcvp_search_a03_h64_reg0001_20260515_202735/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a03-h64-reg0001	+0.5551	+0.2841	+0.4803	+0.3851	+0.4701
dtd	TCVP-FedAdam-a03-h64-reg0001	-0.6481	+2.3843	-0.0362	+1.3100	+0.3310
oxford_pets	TCVP-FedAdam-a03-h64-reg0001	-0.1000	-0.7123	-0.1790	-0.4622	-0.2811
oxford_flowers	TCVP-FedAdam-a03-h64-reg0001	-1.2593	+2.3742	+1.6596	+2.0247	+0.3827
eurosat	TCVP-FedAdam-a03-h64-reg0001	+0.0800	+0.3404	+12.0641	+5.1425	+2.6113
Average	-	-0.2745	+0.9341	+2.7978	+1.6800	+0.7028
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 06: tcvp_search_a03_reg0003 =====
Rationale: Weaken residual regularization; test if reg0001 slightly suppresses novel improvement.
Summary: logs/quick_param_comparison_tcvp_search_a03_reg0003_20260515_211924/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a03-reg0003	+0.1945	-0.1356	+0.6987	+0.2923	+0.2434
dtd	TCVP-FedAdam-a03-reg0003	-1.6204	+2.8009	+0.8333	+1.9339	+0.1567
oxford_pets	TCVP-FedAdam-a03-reg0003	-0.3985	+0.1223	-0.1287	+0.0038	-0.1973
oxford_flowers	TCVP-FedAdam-a03-reg0003	-0.7295	+3.1529	+1.8582	+2.5168	+0.8936
eurosat	TCVP-FedAdam-a03-reg0003	-0.4844	-0.6667	+0.5410	-0.1258	-0.3051
Average	-	-0.6077	+1.0548	+0.7605	+0.9242	+0.1583
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 07: tcvp_search_a03_reg003 =====
Rationale: Strengthen regularization without align loss to isolate L_reg effect.
Summary: logs/quick_param_comparison_tcvp_search_a03_reg003_20260515_221114/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a03-reg003	+0.3207	+0.3873	+0.6659	+0.5307	+0.4256
dtd	TCVP-FedAdam-a03-reg003	-0.6944	+3.0093	-3.1643	+0.1597	-0.2674
oxford_pets	TCVP-FedAdam-a03-reg003	+0.0013	-0.2817	-0.0448	-0.1704	-0.0845
oxford_flowers	TCVP-FedAdam-a03-reg003	-1.3168	+2.8395	+3.5249	+3.1724	+0.9278
eurosat	TCVP-FedAdam-a03-reg003	-0.5555	+3.4285	+2.9462	+3.2137	+1.3291
Average	-	-0.4489	+1.8766	+0.7856	+1.3812	+0.4661
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 08: tcvp_search_a03_detach_reg0001 =====
Rationale: Detach text condition to protect text prompts from MetaNet feedback.
Summary: logs/quick_param_comparison_tcvp_search_a03_detach_reg0001_20260515_230318/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a03-detach-reg0001	+0.3207	+0.2841	+0.9170	+0.6091	+0.4649
dtd	TCVP-FedAdam-a03-detach-reg0001	-1.2963	+3.0787	+0.7488	+2.0472	+0.3754
oxford_pets	TCVP-FedAdam-a03-detach-reg0001	-0.2959	+0.6540	+0.1062	+0.3953	+0.0498
oxford_flowers	TCVP-FedAdam-a03-detach-reg0001	+0.4120	+2.2697	+2.4752	+2.3700	+1.3910
eurosat	TCVP-FedAdam-a03-detach-reg0001	-0.4711	+6.1261	-4.5590	+0.7338	+0.1313
Average	-	-0.2661	+2.4825	-0.0624	+1.2311	+0.4825
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 09: tcvp_search_a03_reg0001_slowlr =====
Rationale: Lower server FedAdam LR to smooth MetaNet global updates.
Summary: logs/quick_param_comparison_tcvp_search_a03_reg0001_slowlr_20260515_235526/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a03-reg0001-slowlr	+0.4243	+0.1033	+0.7096	+0.4147	+0.4194
dtd	TCVP-FedAdam-a03-reg0001-slowlr	-1.1111	+2.5232	-0.1208	+1.3467	+0.1178
oxford_pets	TCVP-FedAdam-a03-reg0001-slowlr	-0.5473	+0.7709	+0.1286	+0.4674	-0.0399
oxford_flowers	TCVP-FedAdam-a03-reg0001-slowlr	-0.1849	+1.0731	+2.2199	+1.6273	+0.7212
eurosat	TCVP-FedAdam-a03-reg0001-slowlr	-0.1377	+2.5833	+4.7718	+3.5690	+1.7157
Average	-	-0.3113	+1.4108	+1.5418	+1.4850	+0.5868
Analysis: CM/HM both improve; candidate setting.
\n===== Iteration 10: tcvp_search_a03_reg0001_groupbind_slot =====
Rationale: Combine TCVP with group-bound selection and slot aggregation to preserve text/vision prompt pairing.
Summary: logs/quick_param_comparison_tcvp_search_a03_reg0001_groupbind_slot_20260516_004720/summary.md

Metrics delta vs baseline:
Dataset	Method	dLocal	dBase	dNovel	dHM	dCM
caltech101	TCVP-FedAdam-a03-reg0001-groupbind-slot	+0.3992	+0.2970	+0.7314	+0.5203	+0.4597
dtd	TCVP-FedAdam-a03-reg0001-groupbind-slot	-0.2315	+0.9954	-1.5580	-0.1423	-0.1869
oxford_pets	TCVP-FedAdam-a03-reg0001-groupbind-slot	-0.0995	-1.2599	-0.1342	-0.7337	-0.4166
oxford_flowers	TCVP-FedAdam-a03-reg0001-groupbind-slot	-0.2651	+1.5954	+1.9291	+1.7578	+0.7464
eurosat	TCVP-FedAdam-a03-reg0001-groupbind-slot	-1.0289	+5.7476	+5.7872	+5.7768	+2.3740
Average	-	-0.2452	+1.4751	+1.3511	+1.4358	+0.5953
Analysis: CM/HM both improve; candidate setting.
\nAll iterations complete. Master log: logs/tcvp_10iter_param_search_20260515_170121.md

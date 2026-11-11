"""Shared CLI for PFL train / eval."""


def add_encoder_args(parser):
    parser.add_argument("--intent_condition", action="store_true",
                        help="condition flow on maneuver probabilities predicted from history only")
    parser.add_argument("--relational_gate", action="store_true",
                        help="modulate relational features using observed geometry and predicted intent")
    parser.add_argument("--use_cuda", action="store_false", default=True)
    parser.add_argument("--use_planning", action="store_false", default=True)
    parser.add_argument("--use_fusion", action="store_true", default=False)
    parser.add_argument("--use_attention", action="store_false", default=True)
    parser.add_argument("--use_relational", action="store_false", default=True,
                        help="ego plan as a per-agent relational feature (needs --use_planning)")
    parser.add_argument("--temporal_relational", action="store_true",
                        help="also inject the unpooled horizon-aligned ego-target relation "
                             "sequence directly into the trajectory vector field")
    parser.add_argument("--temporal_only", action="store_true",
                        help="use the horizon-aligned relation sequence without adding the "
                             "pooled relational embedding to target dynamics")
    parser.add_argument("--temporal_relation_zero", action="store_true",
                        help="capacity control: retain relation modules but replace their "
                             "normalized relation input by zero")
    parser.add_argument("--relation_cross_attention", action="store_true",
                        help="inject relation tokens through a zero-initialized cross-attention "
                             "adapter in every vector-field residual block")
    parser.add_argument("--relation_attention_heads", type=int, default=4)
    parser.add_argument("--mode_conditioned", action="store_true",
                        help="condition each full-trajectory flow sample on a maneuver mode")
    parser.add_argument("--relation_control", choices=["none", "target_cv"], default="none",
                        help="optional capacity-matched non-relational control for ablation only")
    parser.add_argument("--cv_lateral", action="store_true", default=False,
                        help="also extrapolate lateral velocity in the CV baseline "
                             "(off: NGSIM lateral finite differences are mostly noise)")
    parser.add_argument("--train_output_flag", action="store_false", default=True)
    parser.add_argument("--grid_size", type=int, nargs=2, default=[25, 5])
    parser.add_argument("--in_length", type=int, default=16)
    parser.add_argument("--out_length", type=int, default=25)
    parser.add_argument("--num_lat_classes", type=int, default=3)
    parser.add_argument("--num_lon_classes", type=int, default=2)
    parser.add_argument("--temporal_embedding_size", type=int, default=32)
    parser.add_argument("--encoder_size", type=int, default=64)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--decoder_size", type=int, default=128)
    parser.add_argument("--soc_conv_depth", type=int, default=64)
    parser.add_argument("--soc_conv2_depth", type=int, default=16)
    parser.add_argument("--dynamics_encoding_size", type=int, default=32)
    parser.add_argument("--social_context_size", type=int, default=80)
    parser.add_argument("--fuse_enc_size", type=int, default=112)
    parser.add_argument("--dest_dec_size", type=int, nargs="*", default=[128, 64, 32])
    parser.add_argument("--dest_latent_size", type=int, nargs="*", default=[8, 50])
    parser.add_argument("--dest_enc_size", type=int, nargs="*", default=[8, 16])
    parser.add_argument("--zdim", type=int, default=16)
    parser.add_argument("--fdim", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1.3)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--best_of_n", type=int, default=6)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--separate_encoders", action="store_true",
                        help="give the mean head its own encoder, so the regression and flow "
                             "objectives never share parameters")
    parser.add_argument("--use_mean_head", action="store_true",
                        help="construct the legacy deterministic mean head (off for fair PFL experiments)")
    parser.add_argument("--vf_dropout", type=float, default=0.1,
                        help="dropout on the vector field residual branches")
    parser.add_argument("--vf_channels", type=int, nargs="*", default=[64, 128, 256],
                        help="U-Net channel widths per level; controls decoder capacity")
    parser.add_argument("--recon_weight", type=float, default=0.0,
                        help="weight of the one-step endpoint reconstruction term; it is "
                             "algebraically a (1-t)^2 reweighting of the flow-matching residual")
    parser.add_argument("--flow_weight", type=float, default=1.0,
                        help="weight of the flow-matching loss; 0 trains a pure regressor")
    parser.add_argument("--mean_weight", type=float, default=0.0,
                        help="weight of the deterministic mean head's regression loss (0 = head unused)")
    parser.add_argument("--mean_hidden", type=int, default=256)
    parser.add_argument("--freeze_vf", action="store_true",
                        help="train the mean head only, leaving the flow decoder untouched")
    parser.add_argument("--loss_time_weight", type=str, default="none",
                        choices=["none", "linear", "std"],
                        help="per-horizon-step loss weight: linear ramps with the step, "
                             "std uses the fitted residual variance so the loss tracks RMSE in feet")
    return parser


def add_train_args(parser):
    add_encoder_args(parser)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--fixed_validation", action="store_true")
    parser.add_argument("--deterministic", action="store_true",
                        help="request deterministic PyTorch kernels where available")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--name", type=str, default="ngsim_pfl")
    parser.add_argument("--train_set", type=str, required=True)
    parser.add_argument("--val_set", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=500,
                        help="training batches between loss/progress log lines")
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--init_from", type=str, default="",
                        help="checkpoint to initialise weights from, without its optimiser state")
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--val_batches", type=int, default=10)
    parser.add_argument("--val_samples", type=int, default=20,
                        help="flow samples averaged by the preregistered validation metric")
    parser.add_argument("--epoch_validation", action="store_true",
                        help="select checkpoints once per epoch on fixed sample-mean RMSE")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="epoch validations without improvement before stopping (0 disables)")
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--aux_weight", type=float, default=0.3,
                        help="weight of the auxiliary maneuver cross entropy")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--stats_batches", type=int, default=300,
                        help="batches used to fit the residual normalisation")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="AdamW decay, applied to conv/linear weights only")
    return parser


def add_eval_args(parser):
    add_encoder_args(parser)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--name", type=str, default="ngsim_pfl")
    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--plan_info_ds", type=int, default=1)
    parser.add_argument("--mean_k", type=int, default=20,
                        help="samples averaged into the E[Y|c] estimator used for RMSE")
    parser.add_argument("--single_sample", action="store_true",
                        help="report RMSE from one raw sample (shows the sqrt(2) penalty)")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="stop after this many batches (0 = whole test set)")
    parser.add_argument("--cond_draws", type=int, default=0,
                        help="draws used for the endpoint-conditioned mean (0 = off)")
    parser.add_argument("--cond_beta", type=str, default="",
                        help="comma-separated kernel bandwidths, in units of endpoint spread")
    return parser

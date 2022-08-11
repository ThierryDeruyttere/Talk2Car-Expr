import os
os.environ["TORCH_HOME"] = "pretrained_downloads"
os.environ["WANDB_DIR"] = "."
import argparse
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from reinforcer_w_dataset_v2 import Reinforcer
from vocabulary import Vocabulary, KeywordVocabulary
from expr_dataset import VGDataset, T2CDataset
import glob
from torch import load as ckpt_load


def main():
    # fixing the settings for logging
    logging.getLogger("lightning").setLevel(logging.INFO)
    if not args.use_attr_att and args.use_class_att:
        logging.info('Incorrect combination of settings. '
                     'We cannot use the class in attention, '
                     'if there is not attribute attention')
        exit(0)
    # make sure everything is as deterministic as possible
    seed_everything(args.seed)

    if args.old_attr:
        multi_f = False
    else:
        multi_f = True
    # load the different vocabularies
    vocab = Vocabulary(emb_file=args.emb_file, dataset=args.dataset)
    # if args.correct_sort:
    #     attr_vocab = KeywordVocabulary(emb_file=args.emb_file, dataset=args.dataset,
    #                                    vocab_name='correct_sort_keyword_vocab', multi_file=multi_f)
    # else:
    attr_vocab = KeywordVocabulary(emb_file=args.emb_file, dataset=args.dataset, multi_file=multi_f)
    class_vocab = KeywordVocabulary(emb_file=args.emb_file, dataset=args.dataset, data='classes')

    # create the model
    if args.TEST:
        assert args.resume_run_dir is not None, 'When testing the resume_run_dir should be set'
        files = glob.glob(os.path.join(args.resume_run_dir, 'epoch*.ckpt'))
        checkpoint = ckpt_load(files[0])
        hp = checkpoint['hyper_parameters']
        hp['resume_run_dir'] = args.resume_run_dir
        model = Reinforcer.load_from_checkpoint(files[0], vocab=vocab, attr_vocab=attr_vocab,
                                                 class_vocab=class_vocab, hparams=hp)
        logger = None
        checkpoint_callback = None
    else:
        model = Reinforcer(vocab, attr_vocab=attr_vocab, class_vocab=class_vocab, hparams=args)
        # setup the logger
        name = ['{}'.format(args.dataset)]
        if args.DEBUG:
            name = ['DEBUGGING_{}'.format(args.dataset)]
        if args.use_bert:
            name.append('BERT')
        if args.multi_task:
            name.append('MultiTask')
        if args.use_img:
            name.append('Img')
        if args.use_box:
            name.append('Box')
        if args.use_attr_att:
            name.append('AttrAtt')
        if args.use_attr_hot == 'True':
            name.append('AttrHot')
        elif args.use_attr_hot == 'map':
            name.append('AttrMap')
        if args.use_class_att:
            name.append('ClsAtt')
        if args.use_class_mod:
            name.append('Cls')
        if args.use_class_hot == 'True':
            name.append('ClsHot')
        elif args.use_class_hot == 'map':
            name.append('ClsMap')
        if args.use_count:
            name.append('Count')
        if args.use_diff:
            name.append('Diff')
        if args.use_force_attr:
            name.append('Switch{}-{}'.format(args.switch_loss_weight, args.switch_loss))
        if args.attribute_gt:
            name.append('AttGT')
        if args.attribute_prob2hot:
            name.append('AttP2H')
        if args.mmi_loss_weight > 0:
            name.append('MMI{}M{}'.format(args.mmi_loss_weight, args.mmi_loss_margin))
        if args.use_all_negatives:
            name.append("AllNegs")
        if args.lr:
            name.append("lr{}".format(args.lr))
        if len(name) == 1:
            name.append('BASELiNE')
        if args.prepend_name is not None:
            name = [args.prepend_name] + name
        name += ['NumAttr{}'.format(args.max_attrs),
                 'Feat{}'.format(args.feature_hidden),
                 'LSTM{}'.format(args.lstm_hidden)]
        if args.use_attr_hot == 'map' or args.use_class_hot == 'map':
            name.append('KeyMap{}'.format(args.keywordmap_hidden))
        if args.use_attr_att:
            name.append('Att{}'.format(args.attention_hidden))
        name = '_'.join(name)
        save_dir = os.path.join(args.out_dir, 'lightning_logs')
        os.makedirs(save_dir, exist_ok=True)
        if args.logger_type == 'wandb':
            logger = WandbLogger(name=name,
                                 save_dir=save_dir,
                                 project='attr_expr_gen',
                                 )
            # optional: log model topology
            logger.watch(model)
            log_dir = os.path.join(logger.save_dir, logger.experiment.name, logger.experiment.dir.split('/')[-2])
            logger.experiment.config.log_dir = log_dir
        elif args.logger_type == 'tb':
            logger = TensorBoardLogger(save_dir,
                                       name=name)
            log_dir = logger.log_dir
        else:
            logging.warning("NO KNOWN LOGGER SELECTED!")
            logger = None
            log_dir = 'logs'
        os.makedirs(os.path.join(log_dir, 'predictions'), exist_ok=True)

        # create the checkpoint callback
        checkpoint_callback = ModelCheckpoint(filepath=os.path.join(log_dir,
                                                                    '{{epoch:03d}}-{{{}:.2f}}'.format(args.monitor)),
                                              save_last=True,
                                              save_top_k=1,
                                              verbose=True,
                                              monitor="val_acc",
                                              mode='max',
                                              prefix='')


    # create the trainer with all the hparams
    trainer = Trainer.from_argparse_args(args,
                                         checkpoint_callback=checkpoint_callback,
                                         logger=logger,
                                         num_sanity_val_steps=3)

    # train the model
    if not args.TEST:
        trainer.fit(model)
    else:
        trainer.test(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model for generationg bbox expressions')
    parser = Trainer.add_argparse_args(parser)
    # add PROGRAM level args
    parser.add_argument('--DEBUG', action='store_true', help='random seed')
    parser.add_argument('--TEST', action='store_true', help='random seed')
    parser.add_argument('--dataset', type=str, required=True, choices=['t2c', 'vg'])
    parser.add_argument('--logger_type', type=str, default='wandb', choices=['wandb', 'tb', 'none'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--emb_file', type=str,
                        default='/cw/liir/NoCsBack/testliir/datasets/embeddings/glove.840B.300d.txt',
                        help='path to embeddings file (only supports Glove)')
    parser.add_argument('--batch_size_train', type=int, default=16, help='batch size for training')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--num_workers', type=int, default=12, help='how many processes are preparing data')
    parser.add_argument('--monitor', type=str, default='meteor', choices=['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
                                                                          'cider', 'rouge_l', 'meteor', 'spice'],
                        help='what is used to keep track of improvement')
    parser.add_argument('--resume_run_dir', type=str, help='If you want to load a pretrained model to continue or test,'
                                                           'give the dir where the best epoch is stored. '
                                                           'ONLY USED DURING TESTING')
    parser.add_argument('--out_dir', type=str,
                        default='/home2/NoCsBack/hci/thierry/attr_expr/', help='where we export al out data')
    parser.add_argument('--prepend_name', type=str, help='optional addition to name')

    # THIS LINE IS KEY TO PULL THE MODEL NAME AND DATASET NAME
    temp_args, _ = parser.parse_known_args()

    # let the dataset add arguments
    if temp_args.dataset == 'vg':
        parser = VGDataset.add_dataset_specific_args(parser)
    elif temp_args.dataset == 't2c':
        parser = T2CDataset.add_dataset_specific_args(parser)
    else:
        exit('unknown dataset')

    # let the model add arguments
    parser = Reinforcer.add_model_specific_args(parser)
    args = parser.parse_args()
    main()

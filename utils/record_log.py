from prettytable import PrettyTable


class record_log():
    def __init__(self, args):
        self.args = args

    def record_args(self, datas, total_paramters, GLOBAL_SEED):
        with open(self.args.savedir + 'args.txt', 'w') as f:
            f.write('mean:{}\nstd:{}\nstd:{}\n'.format(datas['mean'], datas['std'], datas['classWeights']))
            f.write("Parameters: {} Seed: {}\n".format(str(total_paramters), GLOBAL_SEED))
            f.write(str(self.args))

    def initial_logfile(self):
        logFileLoc = self.args.savedir + self.args.logFile
        logger = open(logFileLoc, 'w')
        logger.write("%s\t%s\t\t%s\t%s\t%s\t%s\t%s\t%s\t\t%s\t\t%s\n" % (
            'Epoch', '   lr', 'Loss(Tr)', 'Loss(Val)', 'FWIOU(Val)', 'mIOU(Val)',  'Pa(Val)', '     Mpa(Val)',
            'PerMiou_set(Val)','    Cpa_set(Val)'))
        return logger

    def resume_logfile(self):
        logFileLoc = self.args.savedir + self.args.logFile
        logger_recored = open(logFileLoc, 'r')
        next(logger_recored)
        lines = logger_recored.readlines()
        logger_recored.close()
        logger = open(logFileLoc, 'a+')
        return logger, lines

    def record_trainVal_log(self, logger, epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, PerMiou_set, Pa_Val, Mpa_Val,
                            Cpa_set, class_dict_df):
        logger.write("%d\t%.6f\t%.4f\t\t%.4f\t\t%0.4f\t\t%0.4f\t\t%0.4f\t    %0.4f\t%s    %s\n" % (
            epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, Pa_Val, Mpa_Val, PerMiou_set, Cpa_set))
        logger.flush()
        print(
            "Epoch  %d  lr=%.6f\tTrain Loss=%.4f      Val Loss=%.4f\t   FWIOU(val)=%.4f\t  Pa(Val)=%.4f\t"  
             % (epoch, lr, lossTr, val_loss, FWIoU, Pa_Val))

        t = PrettyTable(['label_index', 'class_name', 'class_iou', "class_pa"])
        for index in range(class_dict_df.shape[0]):
            t.add_row([class_dict_df['label_index'][index], class_dict_df['class_name'][index], PerMiou_set[index],
                       Cpa_set[index]])
        print(t.get_string(title="Miou is {:.4f}   Mpa is {:.4f}".format(mIOU_val, Mpa_Val)))

    def record_train_log(self, logger, epoch, lr, lossTr):
        logger.write("%d\t%.6f\t%.4f\n" % (epoch, lr, lossTr))
        logger.flush()
        print("Epoch  %d\tlr=%.6f\tTrain Loss=%.4f\n" % (epoch, lr, lossTr))

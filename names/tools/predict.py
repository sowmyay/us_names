# import gcNB.tools.train as gcnb_train
import ngRNN.tools.predict as ngrnn_predict
import gcRNN.tools.predict as gcrnn_predict


def main(args):
    if args.model_type == "generate-rnn":
        ngrnn_predict.main(args)
    else:
        gcrnn_predict.main(args)
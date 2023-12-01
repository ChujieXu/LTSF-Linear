from data_provider.data_loader import Global_Wind_Temp
from torch.utils.data import DataLoader


def data_provider(args, flag):
    Data = Global_Wind_Temp

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader

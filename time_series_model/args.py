import argparse


def Model_args():
    parser = argparse.ArgumentParser()    # 創立parser物件
    parser.add_argument('--epochs', type=int, default=1000, help='訓練回合數')
    parser.add_argument('--batch_size', type=int, default=30, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=20, help='時序長度')
    # 調整-優化器
    parser.add_argument('--optimizer', type=str, default='adam', help='優化器種類')
    # ------ReduceLROnPlateau (loss不再下降才調整學習率)
    parser.add_argument('--factor', type=float, default=0.5, help='學習率每次降低多少')
    parser.add_argument('--patience', type=int, default=5, help='幾次不下降更新學習率')
    # ------StepLR (固定回合調整學習率)
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='權值衰減(防止過擬合)')
    # 調整-模型參數
    parser.add_argument('--input_size', type=int, default=7, help='input_size')    # 輸入_多變量=N
    parser.add_argument('--output_size', type=int, default=1, help='output_size')    # 輸出_多天預測=N
    # ------LSTM
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=2, help='num_layers')

    args = parser.parse_known_args()[0]
    return args

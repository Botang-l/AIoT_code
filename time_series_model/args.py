import argparse


def Model_args():
    parser = argparse.ArgumentParser()    # 創立parser物件
    parser.add_argument('--epochs', type=int, default=10000, help='訓練回合數')
    parser.add_argument('--batch_size', type=int, default=100, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=30, help='時序長度')
    # 調整-優化器
    parser.add_argument('--optimizer', type=str, default='adam', help='優化器種類')
    # ------ReduceLROnPlateau (loss不再下降才調整學習率)
    parser.add_argument('--factor', type=float, default=0.9, help='學習率每次降低多少')
    parser.add_argument('--patience', type=int, default=19, help='幾次不下降更新學習率')
    # ------StepLR (固定回合調整學習率)
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='權值衰減(防止過擬合)')
    # 調整-模型參數
    parser.add_argument('--input_size', type=int, default=15, help='input_size')    # 輸入_多變量=N
    parser.add_argument('--output_size', type=int, default=1, help='output_size')    # 輸出_多天預測=N
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    # ------Transformer
    parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
    args = parser.parse_known_args()[0]
    return args

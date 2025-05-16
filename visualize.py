import matplotlib.pyplot as plt
import numpy as np

def draw_accuracy(baseline, false_10, false_20, perturbation, shuffle):
    """
    시각화 함수
    :param baseline: baseline accuracy
    :param false_10: 10% false accuracy
    :param false_20: 20% false accuracy
    :param perturbation: perturbation accuracy
    :param shuffle: shuffle accuracy
    """
    # 막대 그래프 그리기
    labels = ['Simple CNN', 'VGG', 'ResNet']

    x = np.arange(len(labels))  # 0,1,2
    width = 0.15  # 막대 너비

    fig, ax = plt.subplots(figsize=(8,5))

    rects1 = ax.bar(x - 2*width, baseline, width, label='Baseline')
    rects2 = ax.bar(x - width, false_10, width, label='10% False')
    rects3 = ax.bar(x, false_20, width, label='20% False')
    rects4 = ax.bar(x + width, perturbation, width, label='Perturbation')
    rects5 = ax.bar(x + 2*width, shuffle, width, label='Shuffle')

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Model Accuracy under Different Conditions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0,3),
                        textcoords='offset points',
                        ha='center', va='bottom')

    for rects in [rects1, rects2, rects3, rects4, rects5]:
        autolabel(rects)

    plt.tight_layout()
    plt.show()

from Train_Model import train_model
import matplotlib.pyplot as plt
import seaborn as sns


def plot_graphs():
    
        # set up figure
        fig, ax = plt.subplots(figsize=(8,5))
    
        # plot rewards
        rewards = train_model()
        sns.scatterplot(x=range(len(rewards)), y=rewards, ax=ax)
    
        # set up labels
        ax.set_title('Reward per Episode', fontsize=20)
        ax.set_xlabel('Episode', fontsize=16)
        ax.set_ylabel('Reward', fontsize=16)
    
        # show plot
        plt.show()


if __name__ == "__main__":
    plot_graphs()

from Train_Model import train_model
import matplotlib.pyplot as plt
import seaborn as sns


def plot_graph(ax, title, learning_rate = 0.9, discount_rate = 0.8):

        # plot rewards
        rewards = train_model(learning_rate,discount_rate)
        sns.scatterplot(x=range(len(rewards)), y=rewards, ax=ax, s=2, color='black', legend=False, linewidth = 0)

        # set up labels
        ax.set_title(title, fontsize=12, y=0.7, pad=-14)
        ax.set_xlabel('Episode', fontsize=6)
        ax.set_ylabel('Reward', fontsize=6)
        ax.tick_params(axis='both', which='both', labelsize=6)

def plot_alpha_graphs():
        
        sns.set_theme()
        sns.set(style="ticks")
        sns.set_style("darkgrid")

        mutiplication_factors = [10, 1, 0.1, 0.01]
        learning_rates = [0.1, 0.3, 0.5, 0.7]

        fig, ax = plt.subplots(len(mutiplication_factors), len(learning_rates), figsize=(18,18))
        

        for i in range(len(mutiplication_factors)):
            for j in range(len(learning_rates)):
                plot_graph(ax[i][j], f"\u03B1 = {round(learning_rates[j]*mutiplication_factors[i], 5)}", mutiplication_factors[i]*learning_rates[j])

        # show plots
        plt.tight_layout(pad=5.0)
        plt.show()

plot_alpha_graphs()

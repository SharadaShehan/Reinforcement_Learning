from Train_Model import train_model
import matplotlib.pyplot as plt
import seaborn as sns


def plot_graph(ax, title, learning_rate = 0.07, discount_rate = 0.8):

        # plot rewards
        rewards = train_model(learning_rate,discount_rate)
        sns.scatterplot(x=range(len(rewards)), y=rewards, ax=ax, s=2, color='black', legend=False, linewidth = 0)

        # set up labels
        ax.set_title(title, fontsize=12, y=0.7, pad=-14)
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.tick_params(axis='both', which='both', labelsize=6)

def plot_alpha_graphs():
        
        sns.set_theme()
        sns.set(style="ticks")
        sns.set_style("darkgrid")

        mutiplication_factors = [1, 0.1, 0.01]
        base_learning_rates = [0.1, 0.3, 0.5, 0.7]

        fig, ax = plt.subplots(len(mutiplication_factors), len(base_learning_rates), figsize=(18,18))
        

        for i in range(len(mutiplication_factors)):
            for j in range(len(base_learning_rates)):
                plot_graph(ax[i][j], f"\u03B1 = {round(base_learning_rates[j]*mutiplication_factors[i], 5)}", learning_rate = mutiplication_factors[i]*base_learning_rates[j])

        plt.suptitle('Scatter plots of Reward Values against Number of Episodes for Different Learning Rates', fontsize=20)

        # show plots
        plt.tight_layout(pad=5.0)
        plt.show()

def plot_gamma_graphs():
                
        sns.set_theme()
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        
        gammas = [ [0.4, 0.6, 0.8, 0.9], [0.99, 0.999, 0.9999, 0.99999] ]
        
        fig, ax = plt.subplots(len(gammas), len(gammas[0]), figsize=(16,10))
        
        for i in range(len(gammas)):
            for j in range(len(gammas[0])):
                plot_graph(ax[i][j], f"\u03B3 = {round(gammas[i][j], 5)}", discount_rate = gammas[i][j])
        
        plt.suptitle('Scatter plots of Reward Values against Number of Episodes for Different Discount Rates', fontsize=20)
        
        # show plots
        plt.tight_layout(pad=4.0)
        plt.show()

def plot_alpha_against_gamma_graphs():
        
        sns.set_theme()
        sns.set(style="ticks")
        sns.set_style("darkgrid")
        
        learning_rates = [0.5, 0.05, 0.005]
        gammas = [0.8, 0.9, 0.99, 0.999]
        
        fig, ax = plt.subplots(len(learning_rates), len(gammas), figsize=(18,18))

        for i in range(len(learning_rates)):
            for j in range(len(gammas)):
                plot_graph(ax[i][j], f"\u03B1 = {round(learning_rates[i], 5)}  \u03B3 = {round(gammas[j], 5)}", learning_rate = learning_rates[i], discount_rate = gammas[j])
        
        plt.suptitle('Scatter plots of Reward Values against Number of Episodes for Different Learning Rates and Discount Rates', fontsize=20)
        
        # show plots
        plt.tight_layout(pad=5.0)
        plt.show()


if __name__ == "__main__":
        # plot_alpha_graphs()
        # plot_gamma_graphs()
        plot_alpha_against_gamma_graphs()


import os
import yaml
import glob
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    my_path = os.getcwd()
    dataset = 'emnist'
    iidness = 'noniid'
    avg_type = 'avg_n_samples'
    epochs = 10

    full_path = f'{my_path}/{dataset}/{iidness}/{avg_type}/{epochs}/*/*.yml'
    summary_path = f'{my_path}/{dataset}/{iidness}/{avg_type}/{epochs}/'
    files = glob.glob(full_path)  # list of all .yaml files in a directory
    number_of_files = len(files)
    avg_train_accuracy, avg_train_loss = [0] * epochs, [0] * epochs
    avg_test_accuracy, avg_test_loss = [0] * epochs, [0] * epochs
    for file in files:
        with open(file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                avg_train_accuracy = [a + b for a, b in zip(avg_train_accuracy, data['train_accuracy'])]
                avg_train_loss = [a + b for a, b in zip(avg_train_loss, data['train_loss'])]
                avg_test_accuracy = [a + b for a, b in zip(avg_test_accuracy, data['test_accuracy_list'])]
                avg_test_loss = [a + b for a, b in zip(avg_test_loss, data['test_loss_list'])]
                num_users = data['num_users']
                frac = data['frac']
                local_bs = data['local_bs']
                local_ep = data['local_ep']
                lr = data['lr']
                optimizer = data['optimizer']
            except yaml.YAMLError as exc:
                print(exc)

    final_avg_train_accuracy = [round(accuracy / number_of_files, 3) for accuracy in avg_train_accuracy]
    final_avg_train_loss = [round(loss / number_of_files, 3) for loss in avg_train_loss]

    final_avg_test_accuracy = [round(accuracy / number_of_files, 3) for accuracy in avg_test_accuracy]
    final_avg_test_loss = [round(loss / number_of_files, 3) for loss in avg_test_loss]

    # summary yaml file with all data and results

    data = dict(
        num_users=num_users,
        frac=frac,
        local_bs=local_bs,
        local_ep=local_ep,
        lr=lr,
        optimizer=optimizer,
        final_avg_train_accuracy=final_avg_train_accuracy,
        final_avg_train_loss=final_avg_train_loss,
        final_avg_test_accuracy=final_avg_test_accuracy,
        final_avg_test_loss=final_avg_test_loss,
    )

    with open(f'{summary_path}/summary.yml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # plot summary results
    matplotlib.use('Agg')
    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(final_avg_train_loss)), final_avg_train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{summary_path}/summary_training_loss.pdf')

    # Plot Training Accuracy vs Communication rounds
    plt.figure()
    plt.title('Training Accuracy vs Communication rounds')
    plt.plot(range(len(final_avg_train_accuracy)), final_avg_train_accuracy, color='k')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{summary_path}/summary_training_accuracy.pdf')

    # Plot Loss curve
    plt.figure()
    plt.title('Testing Loss vs Communication rounds')
    plt.plot(range(len(final_avg_train_loss)), final_avg_train_loss, color='r')
    plt.ylabel('Testing loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{summary_path}/summary_testing_loss.pdf')

    # Plot Testing Accuracy vs Communication rounds
    plt.figure()
    plt.title('Testing Accuracy vs Communication rounds')
    plt.plot(range(len(final_avg_train_accuracy)), final_avg_train_accuracy, color='k')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(f'{summary_path}/summary_testing_accuracy.pdf')


import os
import yaml
import glob
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    my_path = os.getcwd()
    dataset = 'emnist-balanced'
    iidness = 'noniid'
    avg_types = ['avg_n_classes', 'avg_n_samples']
    avg_test_accuracy, avg_n_classes_test_accuracy, avg_n_samples_test_accuracy = [], [], []
    avg_test_loss, avg_n_classes_test_loss, avg_n_samples_test_loss = [], [], []
    epochs = 50
    ratios = [50]
    number_of_classes_of_half_of_user = [1, 2, 3, 4, 6]
    # ratios = [25, 50, 75]
    # number_of_classes_of_half_of_user = [1, 6]
    for ratio in ratios:
        for class_dist in number_of_classes_of_half_of_user:
            write_to_summary_path = f'{my_path}/{dataset}/{iidness}/{epochs}/{class_dist}/{ratio}'
            i = 0
            for avg_type in avg_types:
                read_from_path = f'{my_path}/{dataset}/{iidness}/{avg_type}/{epochs}/{class_dist}/{ratio}/summary.yml'
                files = glob.glob(read_from_path)  # list of all .yaml files in a directory
                with open(files[0], 'r') as stream:
                    try:
                        data = yaml.safe_load(stream)
                        # if i == 0:
                        #     avg_test_accuracy = data['final_avg_test_accuracy']
                        #     avg_test_loss = data['final_avg_test_loss']
                        if i == 0:
                            avg_n_classes_test_accuracy = data['final_avg_test_accuracy']
                            avg_n_classes_test_loss = data['final_avg_test_loss']
                        elif i == 1:
                            avg_n_samples_test_accuracy = data['final_avg_test_accuracy']
                            avg_n_samples_test_loss = data['final_avg_test_loss']

                    except yaml.YAMLError as exc:
                        print(exc)
                # summary yaml file with all data and results

                # plot summary results
                matplotlib.use('Agg')
                # Plot Loss curve
                plt.figure()
                plt.title('Testing Loss vs Communication rounds')
                # plt.plot(range(len(avg_test_loss)), avg_test_loss, color='r', linestyle='solid', marker='2',
                #            label="avg")
                plt.plot(range(len(avg_n_classes_test_loss)), avg_n_classes_test_loss, color='g', linestyle='solid', marker='2',
                         label="avg_n_classes")
                plt.plot(range(len(avg_n_samples_test_loss)), avg_n_samples_test_loss, color='b', linestyle='solid', marker='2',
                         label="avg_n_samples")
                plt.legend(prop={'size': 20})
                plt.ylabel('Testing loss')
                plt.xlabel('Communication Rounds')
                plt.savefig(f'{write_to_summary_path}/summary_testing_loss.pdf')

                # Plot Testing Accuracy vs Communication rounds
                plt.figure()
                plt.title('Testing Accuracy vs Communication rounds')
                # plt.plot(range(len(avg_test_accuracy)), avg_test_accuracy, color='r', linestyle='solid', marker='2',
                #          label="avg")
                plt.plot(range(len(avg_n_classes_test_accuracy)), avg_n_classes_test_accuracy, color='g', linestyle='solid', marker='2',
                         label="avg_n_classes")
                plt.plot(range(len(avg_n_samples_test_accuracy)), avg_n_samples_test_accuracy, color='b', linestyle='solid', marker='2',
                         label="avg_n_samples"
                         )
                plt.legend(prop={'size': 20})
                plt.ylabel('Testing Accuracy')
                plt.xlabel('Communication Rounds')
                plt.savefig(f'{write_to_summary_path}/summary_testing_accuracy.pdf')

                i += 1
            # # Plot Loss curve
            # plt.figure()
            # plt.title('Training Loss vs Communication rounds')
            # plt.plot(range(len(final_avg_train_loss)), final_avg_train_loss, color='r')
            # plt.ylabel('Training loss')
            # plt.xlabel('Communication Rounds')
            # plt.savefig(f'{summary_path}/summary_training_loss.pdf')
            #
            # # Plot Training Accuracy vs Communication rounds
            # plt.figure()
            # plt.title('Training Accuracy vs Communication rounds')
            # plt.plot(range(len(final_avg_train_accuracy)), final_avg_train_accuracy, color='k')
            # plt.ylabel('Training Accuracy')
            # plt.xlabel('Communication Rounds')
            # plt.savefig(f'{summary_path}/summary_training_accuracy.pdf')



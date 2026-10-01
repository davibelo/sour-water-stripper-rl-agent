'''
MODELLING - DEEP LEARNING
WITH HYPERPARAMETER OPTIMIZATION
ONE MODEL PER OUTPUT
'''

import json
import os
import logging
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from sklearn.metrics import r2_score

DATA_ID = '04'
MAX_EPOCHS = 1000
PATIENCE = 50
N_TRIALS = 200

LOG_FILE = os.path.join(os.path.dirname(__file__), f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

class FlushableFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    handlers=[FlushableFileHandler(LOG_FILE, mode='w')],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

MODEL_ID = os.path.splitext(os.path.basename(__file__))[0].split('-')[-1]
SCRIPT_FOLDER = os.path.dirname(__file__)
INPUT_FOLDER = SCRIPT_FOLDER
FIGURES_FOLDER = os.path.join(SCRIPT_FOLDER, 'figures')
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, 'output_files')
CHECKPOINT_DIR = os.path.join(SCRIPT_FOLDER, '_optuna_checkpoints')

normalization_path = os.path.join(INPUT_FOLDER, f'sim_results_normalization-{DATA_ID}.json')
with open(normalization_path, 'r', encoding='utf-8') as f:
    normalization = json.load(f)

inputs = normalization['inputs']
outputs = normalization['outputs']

train_path = os.path.join(INPUT_FOLDER, f'sim_results_train-{DATA_ID}.csv')
val_path = os.path.join(INPUT_FOLDER, f'sim_results_val-{DATA_ID}.csv')
test_path = os.path.join(INPUT_FOLDER, f'sim_results_test-{DATA_ID}.csv')

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

x_train_scaled = train_df[inputs].to_numpy(dtype=float)
y_train_scaled = train_df[outputs].to_numpy(dtype=float)

x_val_scaled = val_df[inputs].to_numpy(dtype=float)
y_val_scaled = val_df[outputs].to_numpy(dtype=float)

x_test_scaled = test_df[inputs].to_numpy(dtype=float)
y_test_scaled = test_df[outputs].to_numpy(dtype=float)

x_scaled = np.vstack([x_train_scaled, x_val_scaled, x_test_scaled])
y_scaled = np.vstack([y_train_scaled, y_val_scaled, y_test_scaled])

logging.info(f"x scaled shape: {x_scaled.shape}")
logging.info(f"y scaled shape: {y_scaled.shape}")

logging.info(f"x_train shape: {x_train_scaled.shape}")
logging.info(f"y_train shape: {y_train_scaled.shape}")
logging.info(f"x_val shape: {x_val_scaled.shape}")
logging.info(f"y_val shape: {y_val_scaled.shape}")
logging.info(f"x_test shape: {x_test_scaled.shape}")
logging.info(f"y_test shape: {y_test_scaled.shape}")

num_features = x_train_scaled.shape[1]

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)


def make_objective(y_train_1d, y_val_1d, y_test_1d, output_name):
    def objective(trial):
        tf.keras.backend.clear_session()

        num_dense_layers = trial.suggest_int('num_dense_layers', 1, 5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        activation = trial.suggest_categorical('activation', ['relu', 'swish', 'tanh', 'selu'])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
        l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)

        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(num_features,)))

        for i in range(num_dense_layers):
            neurons = trial.suggest_int(f'neurons_layer_{i}', 16, 256, step=16)
            model.add(layers.Dense(neurons, kernel_regularizer=regularizers.l2(l2_reg)))
            if use_batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(1))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{MODEL_ID}_{output_name}_trial_{trial.number}.keras')

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=PATIENCE, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0
            ),
        ]

        history = model.fit(
            x_train_scaled, y_train_1d,
            validation_data=(x_val_scaled, y_val_1d),
            epochs=MAX_EPOCHS,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        model = tf.keras.models.load_model(checkpoint_path)

        val_loss = model.evaluate(x_val_scaled, y_val_1d, verbose=0)

        y_val_pred = model.predict(x_val_scaled, verbose=0).flatten()
        val_r2 = r2_score(y_val_1d, y_val_pred)

        y_test_pred = model.predict(x_test_scaled, verbose=0).flatten()
        test_r2 = r2_score(y_test_1d, y_test_pred)

        epochs_trained = len(history.history['loss'])

        logging.info(
            f"[{output_name}] Trial {trial.number}: val_loss={val_loss:.6f}, val_R2={val_r2:.4f}, "
            f"test_R2={test_r2:.4f}, epochs={epochs_trained}"
        )

        trial.set_user_attr("val_r2", val_r2)
        trial.set_user_attr("test_r2", test_r2)
        trial.set_user_attr("epochs_trained", epochs_trained)
        trial.set_user_attr("checkpoint_path", checkpoint_path)
        trial.set_user_attr("history", history.history)

        return val_loss

    return objective


def cleanup_checkpoints(study):
    best_path = study.best_trial.user_attrs.get("checkpoint_path")
    for trial in study.trials:
        path = trial.user_attrs.get("checkpoint_path")
        if path and path != best_path and os.path.exists(path):
            os.remove(path)


def trial_callback(study, trial):
    logging.info(
        f"Trial {trial.number} completed with value: {trial.value}. "
        f"Parameters: {trial.params}"
    )


for output_index, output_name in enumerate(outputs):
    logging.info(f"===== Training model for output: {output_name} (index {output_index}) =====")

    y_train_1d = y_train_scaled[:, output_index]
    y_val_1d = y_val_scaled[:, output_index]
    y_test_1d = y_test_scaled[:, output_index]
    y_all_1d = y_scaled[:, output_index]

    study = optuna.create_study(direction='minimize')
    study.optimize(
        make_objective(y_train_1d, y_val_1d, y_test_1d, output_name),
        n_trials=N_TRIALS,
        callbacks=[trial_callback],
    )

    logging.info(f"[{output_name}] Best trial value: {study.best_trial.value}")
    logging.info(f"[{output_name}] Best trial params: {study.best_trial.params}")
    logging.info(f"[{output_name}] Best trial val_R2: {study.best_trial.user_attrs['val_r2']:.4f}")
    logging.info(f"[{output_name}] Best trial test_R2: {study.best_trial.user_attrs['test_r2']:.4f}")
    logging.info(f"[{output_name}] Best trial epochs: {study.best_trial.user_attrs['epochs_trained']}")

    best_checkpoint = study.best_trial.user_attrs["checkpoint_path"]
    best_model = tf.keras.models.load_model(best_checkpoint)
    best_history = study.best_trial.user_attrs["history"]

    best_model_path = os.path.join(OUTPUT_FOLDER, f'{MODEL_ID}_{output_name}_best.keras')
    best_model.save(best_model_path)
    logging.info(f"[{output_name}] Best model saved to {best_model_path}")

    cleanup_checkpoints(study)

    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).write_html(
            os.path.join(FIGURES_FOLDER, f'{MODEL_ID}_{output_name}_optimization_history.html')
        )
        vis.plot_param_importances(study).write_html(
            os.path.join(FIGURES_FOLDER, f'{MODEL_ID}_{output_name}_param_importance.html')
        )
        logging.info(f"[{output_name}] Saved Optuna visualizations.")
    except ImportError:
        logging.warning("Optuna visualization libraries are not installed.")

    y_test_pred = best_model.predict(x_test_scaled, verbose=0).flatten()
    y_all_pred = best_model.predict(x_scaled, verbose=0).flatten()
    residue_test = y_test_1d - y_test_pred
    residue_all = y_all_1d - y_all_pred

    rmse = np.sqrt(best_model.evaluate(x_test_scaled, y_test_1d, batch_size=1000, verbose=0))
    logging.info(f"[{output_name}] RMSE: {rmse}")

    r2_test = r2_score(y_test_1d, y_test_pred)
    logging.info(f"[{output_name}] Test data R2: {r2_test}")

    r2_all = r2_score(y_all_1d, y_all_pred)
    logging.info(f"[{output_name}] All data R2: {r2_all}")

    plt.figure(figsize=(10, 5))
    plt.plot(best_history['loss'], linewidth=2)
    plt.plot(best_history['val_loss'], linewidth=2)
    plt.title(f'Training loss RMSE: {rmse:.3f} ({output_name})', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['train', 'validation'], loc='upper right', fontsize=12)
    plt.grid()
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(os.path.join(FIGURES_FOLDER, f'model_{MODEL_ID}_{output_name}_training_loss.png'))

    prediction_results = np.column_stack((y_test_1d, y_test_pred))
    prediction_results = prediction_results[np.argsort(prediction_results[:, 0])]
    plt.figure(figsize=(10, 5))
    plt.plot(prediction_results[:, 0], linestyle='-', linewidth=0.5, marker='o', markersize=3, label='Real')
    plt.plot(prediction_results[:, 1], linestyle='-', linewidth=0.5, marker='^', markersize=3, label='Prediction')
    plt.title(f'Test Data: y real x y prediction ({output_name}), RMSE: {rmse:.3f}', fontsize=14)
    plt.xlabel('Test Data', fontsize=12)
    plt.ylabel('y normalized', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(os.path.join(FIGURES_FOLDER, f'model_{MODEL_ID}_test_data_real_and_pred_{output_name}.png'))

    plt.figure()
    plt.plot(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50), color='black', label='x = y')
    plt.scatter(y_test_1d, y_test_pred, color='blue', marker='x')
    plt.legend(fontsize=15, loc='best')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predicted Values', fontsize=15)
    plt.title(f'Prediction - Test Data ({output_name})', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f'model_{MODEL_ID}_test_real_vs_pred_{output_name}.png'))

    plt.figure()
    plt.grid(axis='y')
    plt.hist(x=residue_test, bins='auto', ec='black')
    plt.ylabel('Frequency', fontsize=15)
    plt.xlabel('Residue', fontsize=15)
    plt.title(f'Residue - Test Data ({output_name})', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f'model_{MODEL_ID}_test_residue_{output_name}.png'))

    plt.close('all')

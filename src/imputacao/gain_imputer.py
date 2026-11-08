import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class GainImputer:

    def __init__(
        self,
        batch_size=128,
        hint_rate=0.7,
        alpha=50,
        iterations=2000,
        random_state=17
    ):

        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state

        self.sess = None


    # ========================================================
    # XAVIER
    # ========================================================

    def _xavier_init(self, size):

        in_dim = size[0]

        xavier_stddev = (
            1.0 / tf.sqrt(in_dim / 2.0)
        )

        return tf.random_normal(
            shape=size,
            stddev=xavier_stddev
        )


    # ========================================================
    # FIT
    # ========================================================

    def fit(self, X):

        np.random.seed(
            self.random_state
        )

        tf.reset_default_graph()

        tf.set_random_seed(
            self.random_state
        )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        self.no, self.dim = X.shape


        # ----------------------------------------------------
        # Máscara
        #
        # 1 = valor conhecido
        # 0 = valor ausente
        # ----------------------------------------------------

        data_m = (
            1 - np.isnan(X)
        ).astype(np.float32)


        # ====================================================
        # NORMALIZAÇÃO
        # ====================================================

        self.min_val = np.zeros(
            self.dim
        )

        self.max_val = np.zeros(
            self.dim
        )


        norm_data = X.copy()


        for i in range(
            self.dim
        ):

            valores = X[:, i]

            observados = valores[
                ~np.isnan(valores)
            ]


            minimo = np.min(
                observados
            )

            maximo = np.max(
                observados
            )


            self.min_val[i] = minimo

            self.max_val[i] = (
                maximo - minimo
            )


            norm_data[:, i] = (
                norm_data[:, i]
                - minimo
            )


            norm_data[:, i] = (
                norm_data[:, i]
                /
                (
                    self.max_val[i]
                    + 1e-6
                )
            )


        norm_data_x = np.nan_to_num(
            norm_data,
            nan=0.0
        )


        # ====================================================
        # PLACEHOLDERS
        # ====================================================

        self.X_tf = tf.placeholder(
            tf.float32,
            shape=[None, self.dim]
        )

        self.M_tf = tf.placeholder(
            tf.float32,
            shape=[None, self.dim]
        )

        self.H_tf = tf.placeholder(
            tf.float32,
            shape=[None, self.dim]
        )


        h_dim = self.dim


        # ====================================================
        # DISCRIMINADOR
        # ====================================================

        D_W1 = tf.Variable(
            self._xavier_init(
                [self.dim * 2, h_dim]
            )
        )

        D_b1 = tf.Variable(
            tf.zeros(
                shape=[h_dim]
            )
        )


        D_W2 = tf.Variable(
            self._xavier_init(
                [h_dim, h_dim]
            )
        )

        D_b2 = tf.Variable(
            tf.zeros(
                shape=[h_dim]
            )
        )


        D_W3 = tf.Variable(
            self._xavier_init(
                [h_dim, self.dim]
            )
        )

        D_b3 = tf.Variable(
            tf.zeros(
                shape=[self.dim]
            )
        )


        theta_D = [
            D_W1,
            D_W2,
            D_W3,
            D_b1,
            D_b2,
            D_b3
        ]


        # ====================================================
        # GERADOR
        # ====================================================

        G_W1 = tf.Variable(
            self._xavier_init(
                [self.dim * 2, h_dim]
            )
        )

        G_b1 = tf.Variable(
            tf.zeros(
                shape=[h_dim]
            )
        )


        G_W2 = tf.Variable(
            self._xavier_init(
                [h_dim, h_dim]
            )
        )

        G_b2 = tf.Variable(
            tf.zeros(
                shape=[h_dim]
            )
        )


        G_W3 = tf.Variable(
            self._xavier_init(
                [h_dim, self.dim]
            )
        )

        G_b3 = tf.Variable(
            tf.zeros(
                shape=[self.dim]
            )
        )


        theta_G = [
            G_W1,
            G_W2,
            G_W3,
            G_b1,
            G_b2,
            G_b3
        ]


        # ====================================================
        # GERADOR
        # ====================================================

        def generator(x, m):

            inputs = tf.concat(
                values=[x, m],
                axis=1
            )

            G_h1 = tf.nn.relu(
                tf.matmul(
                    inputs,
                    G_W1
                )
                + G_b1
            )

            G_h2 = tf.nn.relu(
                tf.matmul(
                    G_h1,
                    G_W2
                )
                + G_b2
            )

            G_prob = tf.nn.sigmoid(
                tf.matmul(
                    G_h2,
                    G_W3
                )
                + G_b3
            )

            return G_prob


        # ====================================================
        # DISCRIMINADOR
        # ====================================================

        def discriminator(x, h):

            inputs = tf.concat(
                values=[x, h],
                axis=1
            )

            D_h1 = tf.nn.relu(
                tf.matmul(
                    inputs,
                    D_W1
                )
                + D_b1
            )

            D_h2 = tf.nn.relu(
                tf.matmul(
                    D_h1,
                    D_W2
                )
                + D_b2
            )

            D_prob = tf.nn.sigmoid(
                tf.matmul(
                    D_h2,
                    D_W3
                )
                + D_b3
            )

            return D_prob


        # ====================================================
        # REDE
        # ====================================================

        self.G_sample = generator(
            self.X_tf,
            self.M_tf
        )


        Hat_X = (
            self.X_tf
            * self.M_tf
            +
            self.G_sample
            * (1 - self.M_tf)
        )


        D_prob = discriminator(
            Hat_X,
            self.H_tf
        )


        # ====================================================
        # FUNÇÕES DE CUSTO
        # ====================================================

        D_loss = -tf.reduce_mean(

            self.M_tf
            * tf.log(
                D_prob + 1e-8
            )

            +

            (1 - self.M_tf)
            * tf.log(
                1.0 - D_prob + 1e-8
            )
        )


        G_loss_temp = -tf.reduce_mean(

            (1 - self.M_tf)

            * tf.log(
                D_prob + 1e-8
            )
        )


        MSE_loss = tf.reduce_mean(

            (
                self.M_tf
                * self.X_tf
                -
                self.M_tf
                * self.G_sample
            ) ** 2

        ) / tf.reduce_mean(
            self.M_tf
        )


        G_loss = (
            G_loss_temp
            +
            self.alpha
            * MSE_loss
        )


        # ====================================================
        # OTIMIZADORES
        # ====================================================

        D_solver = (
            tf.train.AdamOptimizer()
            .minimize(
                D_loss,
                var_list=theta_D
            )
        )


        G_solver = (
            tf.train.AdamOptimizer()
            .minimize(
                G_loss,
                var_list=theta_G
            )
        )


        # ====================================================
        # SESSÃO
        # ====================================================

        self.sess = tf.Session()


        self.sess.run(
            tf.global_variables_initializer()
        )


        # ====================================================
        # TREINAMENTO
        # ====================================================

        batch_size_real = min(
            self.batch_size,
            self.no
        )


        for _ in range(
            self.iterations
        ):

            batch_idx = np.random.choice(
                self.no,
                batch_size_real,
                replace=False
            )


            X_mb = norm_data_x[
                batch_idx
            ]

            M_mb = data_m[
                batch_idx
            ]


            # -----------------------------------------------
            # Ruído
            # -----------------------------------------------

            Z_mb = np.random.uniform(
                0,
                0.01,
                size=[
                    batch_size_real,
                    self.dim
                ]
            )


            X_mb = (
                M_mb * X_mb
                +
                (1 - M_mb) * Z_mb
            )


            # -----------------------------------------------
            # Hint
            # -----------------------------------------------

            H_mb_temp = (
                np.random.uniform(
                    0,
                    1,
                    size=[
                        batch_size_real,
                        self.dim
                    ]
                )
                < self.hint_rate
            ).astype(
                np.float32
            )


            H_mb = (
                M_mb
                * H_mb_temp
            )


            # -----------------------------------------------
            # Treinar discriminador
            # -----------------------------------------------

            self.sess.run(
                D_solver,
                feed_dict={
                    self.X_tf:
                        X_mb,

                    self.M_tf:
                        M_mb,

                    self.H_tf:
                        H_mb
                }
            )


            # -----------------------------------------------
            # Treinar gerador
            # -----------------------------------------------

            self.sess.run(
                G_solver,
                feed_dict={
                    self.X_tf:
                        X_mb,

                    self.M_tf:
                        M_mb,

                    self.H_tf:
                        H_mb
                }
            )


        return self


    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(self, X):

        X = np.asarray(
            X,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Máscara do teste
        # ----------------------------------------------------

        M = (
            1 - np.isnan(X)
        ).astype(
            np.float32
        )


        # ====================================================
        # NORMALIZAR COM PARÂMETROS DO TREINO
        # ====================================================

        norm_data = X.copy()


        for i in range(
            self.dim
        ):

            norm_data[:, i] = (
                norm_data[:, i]
                - self.min_val[i]
            )


            norm_data[:, i] = (
                norm_data[:, i]
                /
                (
                    self.max_val[i]
                    + 1e-6
                )
            )


        norm_data_x = np.nan_to_num(
            norm_data,
            nan=0.0
        )


        # ====================================================
        # RUÍDO NOS VALORES AUSENTES
        # ====================================================

        Z = np.random.uniform(
            0,
            0.01,
            size=X.shape
        )


        X_input = (
            M * norm_data_x
            +
            (1 - M) * Z
        )


        # ====================================================
        # GERADOR JÁ TREINADO
        # ====================================================

        G_sample = self.sess.run(

            self.G_sample,

            feed_dict={

                self.X_tf:
                    X_input,

                self.M_tf:
                    M
            }
        )


        # ====================================================
        # MANTER VALORES OBSERVADOS
        # ====================================================

        imputed_norm = (
            M * norm_data_x
            +
            (1 - M) * G_sample
        )


        # ====================================================
        # VOLTAR PARA ESCALA ORIGINAL
        # ====================================================

        imputed = (
            imputed_norm
            * (
                self.max_val
                + 1e-6
            )
            + self.min_val
        )


        # ====================================================
        # ARREDONDAMENTO
        # ====================================================

        # Todas as nossas variáveis representam contagens.
        # Ex.: gols, chutes, cartões, escanteios.
        #
        # Portanto, não queremos algo como:
        # HTHG = 1.48372

        imputed = np.round(
            imputed
        )


        return imputed


    # ========================================================
    # FECHAR SESSÃO
    # ========================================================

    def close(self):

        if self.sess is not None:

            self.sess.close()

            self.sess = None
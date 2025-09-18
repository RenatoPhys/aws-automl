"""
Modelo de Double Machine Learning para Análise de Sensibilidade 
de Taxa de Juros em Financiamento Imobiliário usando EconML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# EconML - Importações principais
from econml.dml import (
    LinearDML, 
    SparseLinearDML, 
    NonParamDML,
    CausalForestDML,
    KernelDML
)
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from econml.dr import DRLearner, ForestDRLearner
from econml.inference import BootstrapInference
from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter

# Configuração de estilo para visualizações
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# 1. GERAÇÃO E PREPARAÇÃO DE DADOS
# ==========================================

class DadosFinanciamentoImobiliario:
    """
    Classe para gerar e preparar dados de financiamento imobiliário
    """
    
    def __init__(self, n_samples=5000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
        
    def gerar_dados(self):
        """
        Gera dataset sintético realista de financiamento imobiliário
        """
        n = self.n_samples
        
        # ========== Características do Cliente (X) ==========
        # Demográficas
        idade = np.random.normal(38, 12, n)
        idade = np.clip(idade, 21, 70)
        
        anos_emprego = np.random.exponential(5, n)
        anos_emprego = np.clip(anos_emprego, 0, 40)
        
        # Financeiras
        renda_mensal = np.random.lognormal(8.8, 0.6, n)
        renda_mensal = np.clip(renda_mensal, 2000, 100000)
        
        score_credito = 300 + np.random.beta(5, 3, n) * 550
        score_credito = np.clip(score_credito, 300, 850)
        
        divida_renda_ratio = np.random.beta(2, 5, n) * 0.6
        
        # Histórico
        num_emprestimos_anteriores = np.random.poisson(1.5, n)
        num_emprestimos_anteriores = np.clip(num_emprestimos_anteriores, 0, 10)
        
        conta_poupanca = np.random.lognormal(9, 1.5, n)
        conta_poupanca = np.clip(conta_poupanca, 0, 500000)
        
        # ========== Características do Imóvel ==========
        valor_imovel = np.random.lognormal(12.7, 0.5, n)
        valor_imovel = np.clip(valor_imovel, 80000, 3000000)
        
        entrada_percentual = 0.05 + np.random.beta(2, 5, n) * 0.45
        
        prazo_meses = np.random.choice([120, 180, 240, 300, 360, 420], n, 
                                       p=[0.05, 0.15, 0.30, 0.25, 0.20, 0.05])
        
        # Categóricas
        tipo_imovel = np.random.choice(['Apartamento', 'Casa', 'Cobertura', 'Terreno'], 
                                       n, p=[0.5, 0.35, 0.1, 0.05])
        
        regiao = np.random.choice(['Norte', 'Sul', 'Centro', 'Leste', 'Oeste'], 
                                  n, p=[0.15, 0.20, 0.30, 0.20, 0.15])
        
        novo_usado = np.random.choice(['Novo', 'Usado'], n, p=[0.3, 0.7])
        
        # ========== Variáveis Econômicas (W) ==========
        taxa_selic = np.random.normal(7.5, 1.5, n)
        taxa_selic = np.clip(taxa_selic, 4, 12)
        
        inflacao = np.random.normal(4.5, 1, n)
        inflacao = np.clip(inflacao, 2, 8)
        
        indice_confianca = np.random.normal(100, 15, n)
        
        # ========== Taxa de Juros (T) - Tratamento ==========
        # A taxa depende de características observáveis com alguma aleatoriedade
        taxa_base = 6.5
        
        ajuste_score = -(score_credito - 600) * 0.008
        ajuste_entrada = -(entrada_percentual - 0.20) * 3
        ajuste_prazo = (prazo_meses - 240) * 0.003
        ajuste_divida = divida_renda_ratio * 2
        ajuste_selic = taxa_selic * 0.3
        ruido = np.random.normal(0, 0.8, n)
        
        taxa_juros = (taxa_base + ajuste_score + ajuste_entrada + 
                     ajuste_prazo + ajuste_divida + ajuste_selic + ruido)
        taxa_juros = np.clip(taxa_juros, 3.5, 18)
        
        # ========== Variáveis de Resultado (Y) ==========
        valor_financiamento = valor_imovel * (1 - entrada_percentual)
        
        # 1. Probabilidade de Aprovação
        linear_score = (
            2.5 +
            (score_credito - 500) * 0.008 -
            taxa_juros * 0.25 +
            np.log(renda_mensal/1000) * 0.4 -
            (valor_financiamento/renda_mensal - 30) * 0.03 +
            np.log(conta_poupanca/1000 + 1) * 0.1 -
            divida_renda_ratio * 2 +
            (entrada_percentual - 0.1) * 3
        )
        
        # Adicionar não-linearidades
        nonlinear_effect = (
            -0.01 * (taxa_juros - 8) ** 2 +
            0.005 * (score_credito - 700) * (taxa_juros - 8)
        )
        
        prob_aprovacao = 1 / (1 + np.exp(-(linear_score + nonlinear_effect + 
                                          np.random.normal(0, 0.3, n))))
        
        # 2. Taxa de Inadimplência
        inadimplencia_score = (
            -4 +
            taxa_juros * 0.35 -
            (score_credito - 500) * 0.012 +
            (valor_financiamento/renda_mensal - 30) * 0.08 -
            entrada_percentual * 4 +
            divida_renda_ratio * 3 -
            np.log(conta_poupanca/1000 + 1) * 0.15
        )
        
        # Efeitos heterogêneos
        if_high_risk = (score_credito < 600) * taxa_juros * 0.1
        
        taxa_inadimplencia = 1 / (1 + np.exp(-(inadimplencia_score + if_high_risk +
                                               np.random.normal(0, 0.4, n))))
        
        # 3. Valor Presente Líquido do Empréstimo (NPV)
        receita_juros = valor_financiamento * taxa_juros/100 * prazo_meses/12
        perda_esperada = valor_financiamento * taxa_inadimplencia * 0.5
        custo_capital = valor_financiamento * taxa_selic/100 * prazo_meses/12
        
        npv = receita_juros - perda_esperada - custo_capital + np.random.normal(0, 5000, n)
        
        # ========== Criar DataFrame ==========
        df = pd.DataFrame({
            # Tratamento
            'taxa_juros': taxa_juros,
            
            # Outcomes
            'prob_aprovacao': prob_aprovacao,
            'taxa_inadimplencia': taxa_inadimplencia,
            'npv': npv,
            'valor_financiamento': valor_financiamento,
            
            # Features contínuas
            'idade': idade,
            'anos_emprego': anos_emprego,
            'renda_mensal': renda_mensal,
            'score_credito': score_credito,
            'divida_renda_ratio': divida_renda_ratio,
            'num_emprestimos_anteriores': num_emprestimos_anteriores,
            'conta_poupanca': conta_poupanca,
            'valor_imovel': valor_imovel,
            'entrada_percentual': entrada_percentual,
            'prazo_meses': prazo_meses,
            'taxa_selic': taxa_selic,
            'inflacao': inflacao,
            'indice_confianca': indice_confianca,
            
            # Features categóricas
            'tipo_imovel': tipo_imovel,
            'regiao': regiao,
            'novo_usado': novo_usado
        })
        
        # Criar dummies para categóricas
        df = pd.get_dummies(df, columns=['tipo_imovel', 'regiao', 'novo_usado'], 
                           drop_first=True)
        
        return df
    
    def preparar_dados_dml(self, df, outcome_col):
        """
        Prepara dados no formato esperado pelo EconML
        """
        # Definir colunas
        treatment_col = 'taxa_juros'
        
        # Features (X) - todas exceto tratamento e outcomes
        outcome_cols = ['prob_aprovacao', 'taxa_inadimplencia', 'npv', 'valor_financiamento']
        feature_cols = [col for col in df.columns 
                       if col not in outcome_cols + [treatment_col]]
        
        # Separar variáveis
        Y = df[outcome_col].values.reshape(-1, 1)
        T = df[treatment_col].values.reshape(-1, 1)
        X = df[feature_cols].values
        
        # Criar algumas features de controle (W) - subset importante
        control_cols = ['score_credito', 'renda_mensal', 'entrada_percentual', 
                       'divida_renda_ratio', 'prazo_meses', 'taxa_selic']
        W = df[control_cols].values
        
        # Normalizar features
        scaler_X = StandardScaler()
        scaler_W = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        W_scaled = scaler_W.fit_transform(W)
        
        return Y, T, X_scaled, W_scaled, feature_cols, control_cols

# ==========================================
# 2. MODELOS DML COM ECONML
# ==========================================

class AnaliseDMLCompleta:
    """
    Classe para executar análise completa usando diferentes estimadores DML
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.modelos = {}
        self.resultados = {}
        
    def configurar_modelos_ml(self):
        """
        Configura modelos de ML base para usar no DML
        """
        # Modelo para outcome (Y|X,W)
        model_y = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        # Modelo para tratamento (T|X,W)
        model_t = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            random_state=self.random_state
        )
        
        # Modelo final para efeito heterogêneo
        model_final = LassoCV(cv=5, random_state=self.random_state)
        
        return model_y, model_t, model_final
    
    def executar_linear_dml(self, Y, T, X, W):
        """
        DML Linear - assume efeito linear do tratamento
        """
        print("\n📊 Linear DML")
        print("-" * 50)
        
        model_y, model_t, model_final = self.configurar_modelos_ml()
        
        # Configurar LinearDML
        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            model_final=model_final,
            discrete_treatment=False,
            cv=5,
            random_state=self.random_state
        )
        
        # Fit com bootstrap para inferência
        dml.fit(Y, T, X=X, W=W, inference='bootstrap', inference_args={'n_bootstrap': 100})
        
        # Efeito médio do tratamento (ATE)
        ate = dml.ate(X)
        ate_interval = dml.ate_interval(X, alpha=0.05)
        
        # Efeito constante (CATE médio)
        const_marginal_effect = dml.const_marginal_effect(X)
        const_marginal_interval = dml.const_marginal_effect_interval(X, alpha=0.05)
        
        self.modelos['linear_dml'] = dml
        
        print(f"ATE: {ate:.4f}")
        print(f"IC 95%: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")
        print(f"Efeito Marginal Constante: {const_marginal_effect[0]:.4f}")
        
        return dml
    
    def executar_causal_forest_dml(self, Y, T, X, W):
        """
        Causal Forest DML - permite efeitos não-lineares e heterogêneos
        """
        print("\n🌲 Causal Forest DML")
        print("-" * 50)
        
        model_y, model_t, _ = self.configurar_modelos_ml()
        
        # Configurar CausalForestDML
        dml = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=100,
            min_samples_leaf=10,
            max_depth=None,
            discrete_treatment=False,
            cv=5,
            random_state=self.random_state
        )
        
        # Fit
        dml.fit(Y, T, X=X, W=W, inference='blb')
        
        # Efeitos
        ate = dml.ate(X)
        ate_interval = dml.ate_interval(X, alpha=0.05)
        
        # Feature importance para heterogeneidade
        feature_importance = dml.feature_importances_
        
        self.modelos['causal_forest'] = dml
        
        print(f"ATE: {ate:.4f}")
        print(f"IC 95%: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")
        print(f"Top 3 features para heterogeneidade: {np.argsort(feature_importance)[-3:]}")
        
        return dml
    
    def executar_kernel_dml(self, Y, T, X, W):
        """
        Kernel DML - usa métodos de kernel para flexibilidade
        """
        print("\n🔮 Kernel DML")
        print("-" * 50)
        
        model_y, model_t, _ = self.configurar_modelos_ml()
        
        # Configurar KernelDML
        dml = KernelDML(
            model_y=model_y,
            model_t=model_t,
            bw='silverman',  # bandwidth selection
            discrete_treatment=False,
            cv=5,
            random_state=self.random_state
        )
        
        # Fit
        dml.fit(Y, T, X=X, W=W)
        
        # Efeitos
        effects = dml.effect(X)
        
        self.modelos['kernel_dml'] = dml
        
        print(f"Efeito Médio: {np.mean(effects):.4f}")
        print(f"Desvio Padrão dos Efeitos: {np.std(effects):.4f}")
        
        return dml
    
    def executar_dr_learner(self, Y, T, X, W):
        """
        Doubly Robust Learner - combina modelos de propensity e outcome
        """
        print("\n🛡️ Doubly Robust Learner")
        print("-" * 50)
        
        model_y = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        model_t = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        model_final = RandomForestRegressor(n_estimators=100, max_depth=10)
        
        # Configurar DRLearner
        dr = DRLearner(
            model_regression=model_y,
            model_propensity=model_t,
            model_final=model_final,
            cv=5,
            random_state=self.random_state
        )
        
        # Fit
        dr.fit(Y, T, X=X, W=W, inference='bootstrap')
        
        # Efeitos
        ate = dr.ate(X)
        ate_interval = dr.ate_interval(X, alpha=0.05)
        
        self.modelos['dr_learner'] = dr
        
        print(f"ATE: {ate:.4f}")
        print(f"IC 95%: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")
        
        return dr
    
    def executar_metalearners(self, Y, T, X):
        """
        Meta-learners: S-Learner, T-Learner, X-Learner
        """
        print("\n🔄 Meta-Learners")
        print("-" * 50)
        
        # S-Learner
        s_learner = SLearner(overall_model=RandomForestRegressor(n_estimators=100))
        s_learner.fit(Y, T, X=X)
        s_ate = np.mean(s_learner.effect(X))
        
        # T-Learner  
        t_learner = TLearner(models=RandomForestRegressor(n_estimators=100))
        t_learner.fit(Y, T, X=X)
        t_ate = np.mean(t_learner.effect(X))
        
        # X-Learner
        x_learner = XLearner(
            models=RandomForestRegressor(n_estimators=100),
            propensity_model=RandomForestRegressor(n_estimators=50)
        )
        x_learner.fit(Y, T, X=X)
        x_ate = np.mean(x_learner.effect(X))
        
        self.modelos['s_learner'] = s_learner
        self.modelos['t_learner'] = t_learner
        self.modelos['x_learner'] = x_learner
        
        print(f"S-Learner ATE: {s_ate:.4f}")
        print(f"T-Learner ATE: {t_ate:.4f}")
        print(f"X-Learner ATE: {x_ate:.4f}")
        
        return s_learner, t_learner, x_learner

# ==========================================
# 3. ANÁLISE DE HETEROGENEIDADE E INTERPRETAÇÃO
# ==========================================

class InterpretadorResultados:
    """
    Classe para interpretar e visualizar resultados do DML
    """
    
    def __init__(self, modelo_principal, X, feature_names):
        self.modelo = modelo_principal
        self.X = X
        self.feature_names = feature_names
        
    def analisar_heterogeneidade(self, df_original, var_analise, n_bins=5):
        """
        Analisa como o efeito do tratamento varia com uma variável específica
        """
        # Obter efeitos individuais (CATE)
        cate = self.modelo.effect(self.X)
        
        # Criar bins da variável de análise
        var_values = df_original[var_analise].values
        bins = pd.qcut(var_values, n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])
        
        # Calcular efeito médio por bin
        resultados = pd.DataFrame({
            'Bin': bins,
            'CATE': cate.flatten(),
            var_analise: var_values
        })
        
        efeitos_por_bin = resultados.groupby('Bin')['CATE'].agg(['mean', 'std', 'count'])
        
        return efeitos_por_bin, resultados
    
    def interpretar_com_arvore(self):
        """
        Usa uma árvore de decisão para interpretar o CATE
        """
        # Criar interpretador de árvore única
        interpreter = SingleTreeCateInterpreter(max_depth=5, min_samples_leaf=50)
        
        # Fit no modelo
        interpreter.interpret(self.modelo, self.X)
        
        # Obter feature importances da árvore interpretadora
        feature_importances = interpreter.feature_importances_
        
        # Criar DataFrame com importâncias
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        return interpreter, importance_df
    
    def criar_policy_tree(self, custo_tratamento=0):
        """
        Cria árvore de política ótima para decisão de tratamento
        """
        # Criar interpretador de política
        policy_interpreter = SingleTreePolicyInterpreter(
            max_depth=5,
            min_samples_leaf=50
        )
        
        # Fit considerando custo do tratamento
        policy_interpreter.interpret(self.modelo, self.X, sample_treatment_costs=custo_tratamento)
        
        return policy_interpreter

# ==========================================
# 4. VISUALIZAÇÕES
# ==========================================

def criar_visualizacoes_completas(df, modelos, X, feature_names):
    """
    Cria conjunto completo de visualizações
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Obter CATEs do modelo principal
    modelo_principal = modelos.get('causal_forest', modelos.get('linear_dml'))
    cates = modelo_principal.effect(X).flatten()
    
    # 1. Distribuição dos CATEs
    ax1 = plt.subplot(2, 4, 1)
    ax1.hist(cates, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(cates), color='red', linestyle='--', label=f'ATE={np.mean(cates):.4f}')
    ax1.set_title('Distribuição dos Efeitos Individuais (CATE)')
    ax1.set_xlabel('Efeito do Tratamento')
    ax1.set_ylabel('Frequência')
    ax1.legend()
    
    # 2. CATE vs Taxa de Juros
    ax2 = plt.subplot(2, 4, 2)
    scatter = ax2.scatter(df['taxa_juros'], cates, c=df['score_credito'], 
                         cmap='coolwarm', alpha=0.5, s=20)
    ax2.set_title('Efeito vs Taxa de Juros')
    ax2.set_xlabel('Taxa de Juros (%)')
    ax2.set_ylabel('CATE')
    plt.colorbar(scatter, ax=ax2, label='Score Crédito')
    
    # 3. Heterogeneidade por Score de Crédito
    ax3 = plt.subplot(2, 4, 3)
    score_bins = pd.qcut(df['score_credito'], 5)
    cate_by_score = pd.DataFrame({'score_bin': score_bins, 'cate': cates})
    grouped = cate_by_score.groupby('score_bin')['cate'].mean()
    
    ax3.bar(range(len(grouped)), grouped.values, alpha=0.7)
    ax3.set_xticks(range(len(grouped)))
    ax3.set_xticklabels([f'{int(i.left)}-{int(i.right)}' for i in grouped.index], 
                        rotation=45)
    ax3.set_title('Efeito Médio por Score de Crédito')
    ax3.set_xlabel('Faixa de Score')
    ax3.set_ylabel('CATE Médio')
    
    # 4. Heterogeneidade por Entrada
    ax4 = plt.subplot(2, 4, 4)
    entrada_bins = pd.qcut(df['entrada_percentual'], 4)
    cate_by_entrada = pd.DataFrame({'entrada_bin': entrada_bins, 'cate': cates})
    grouped_entrada = cate_by_entrada.groupby('entrada_bin')['cate'].mean()
    
    ax4.bar(range(len(grouped_entrada)), grouped_entrada.values, alpha=0.7, color='green')
    ax4.set_xticks(range(len(grouped_entrada)))
    ax4.set_xticklabels([f'{i.left:.0%}-{i.right:.0%}' for i in grouped_entrada.index], 
                        rotation=45)
    ax4.set_title('Efeito Médio por % de Entrada')
    ax4.set_xlabel('Percentual de Entrada')
    ax4.set_ylabel('CATE Médio')
    
    # 5. Comparação entre Modelos
    ax5 = plt.subplot(2, 4, 5)
    model_names = []
    model_ates = []
    
    for name, model in modelos.items():
        if hasattr(model, 'effect'):
            model_names.append(name.replace('_', ' ').title())
            effects = model.effect(X).flatten()
            model_ates.append(np.mean(effects))
    
    ax5.barh(model_names, model_ates, alpha=0.7)
    ax5.set_title('Comparação de ATEs entre Modelos')
    ax5.set_xlabel('Average Treatment Effect')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 6. Distribuição do Tratamento
    ax6 = plt.subplot(2, 4, 6)
    ax6.hist(df['taxa_juros'], bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax6.set_title('Distribuição da Taxa de Juros')
    ax6.set_xlabel('Taxa de Juros (%)')
    ax6.set_ylabel('Frequência')
    
    # 7. CATE vs Renda
    ax7 = plt.subplot(2, 4, 7)
    ax7.hexbin(np.log10(df['renda_mensal']), cates, gridsize=25, cmap='YlOrRd')
    ax7.set_title('Efeito vs Renda (log scale)')
    ax7.set_xlabel('Log10(Renda Mensal)')
    ax7.set_ylabel('CATE')
    
    # 8. Box plot de CATEs por Prazo
    ax8 = plt.subplot(2, 4, 8)
    prazo_cats = pd.cut(df['prazo_meses'], bins=[0, 180, 300, 500], 
                        labels=['Curto', 'Médio', 'Longo'])
    cate_df = pd.DataFrame({'prazo': prazo_cats, 'cate': cates})
    
    cate_df.boxplot(column='cate', by='prazo', ax=ax8)
    ax8.set_title('Distribuição de Efeitos por Prazo')
    ax8.set_xlabel('Prazo do Financiamento')
    ax8.set_ylabel('CATE')
    plt.sca(ax8)
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# ==========================================
# 5. PIPELINE PRINCIPAL DE EXECUÇÃO
# ==========================================

def executar_analise_econml():
    """
    Pipeline principal para análise com EconML
    """
    print("=" * 60)
    print("ANÁLISE DML - FINANCIAMENTO IMOBILIÁRIO")
    print("Powered by EconML")
    print("=" * 60)
    
    # 1. Gerar e preparar dados
    print("\n📁 Preparando Dados...")
    gerador = DadosFinanciamentoImobiliario(n_samples=5000)
    df = gerador.gerar_dados()
    print(f"   ✓ {len(df)} observações geradas")
    print(f"   ✓ {len(df.columns)} features")
    
    # 2. Preparar para DML
    Y_aprov, T, X, W, feature_names, control_names = gerador.preparar_dados_dml(
        df, 'prob_aprovacao'
    )
    
    print(f"\n📐 Dimensões dos Dados:")
    print(f"   Y (outcome): {Y_aprov.shape}")
    print(f"   T (treatment): {T.shape}")
    print(f"   X (features): {X.shape}")
    print(f"   W (controls): {W.shape}")
    
    # 3. Executar diferentes modelos DML
    print("\n🚀 Executando Modelos DML...")
    analise = AnaliseDMLCompleta()
    
    # Linear DML
    linear_dml = analise.executar_linear_dml(Y_aprov, T, X, W)
    
    # Causal Forest DML
    cf_dml = analise.executar_causal_forest_dml(Y_aprov, T, X, W)
    
    # Kernel DML
    kernel_dml = analise.executar_kernel_dml(Y_aprov, T, X, W)
    
    # DR Learner
    dr_learner = analise.executar_dr_learner(Y_aprov, T, X, W)
    
    # Meta-learners
    s_learn, t_learn, x_learn = analise.executar_metalearners(Y_aprov, T, X)
    
    # 4. Análise de Heterogeneidade
    print("\n🔍 Análise de Heterogeneidade...")
    interpretador = InterpretadorResultados(cf_dml, X, feature_names)
    
    # Heterogeneidade por Score de Crédito
    efeitos_score, df_het = interpretador.analisar_heterogeneidade(
        df, 'score_credito', n_bins=5
    )
    
    print("\nEfeitos por Quintil de Score de Crédito:")
    print(efeitos_score)
    
    # 5. Interpretação com Árvore
    print("\n🌳 Interpretação com Árvore de Decisão...")
    tree_interpreter, importance_df = interpretador.interpretar_com_arvore()
    
    print("\nTop 10 Features mais importantes para heterogeneidade:")
    print(importance_df.head(10))
    
    # 6. Criar Policy Tree
    print("\n📋 Árvore de Política Ótima...")
    policy_tree = interpretador.criar_policy_tree(custo_tratamento=0.01)
    
    # 7. Análise para Taxa de Inadimplência
    print("\n💰 Análise Adicional: Taxa de Inadimplência")
    Y_inad, _, _, _, _, _ = gerador.preparar_dados_dml(df, 'taxa_inadimplencia')
    
    dml_inad = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100),
        model_t=RandomForestRegressor(n_estimators=100),
        discrete_treatment=False,
        cv=5
    )
    dml_inad.fit(Y_inad, T, X=X, W=W, inference='bootstrap')
    
    ate_inad = dml_inad.ate(X)
    ate_interval_inad = dml_inad.ate_interval(X, alpha=0.05)
    
    print(f"   ATE sobre Inadimplência: {ate_inad:.4f}")
    print(f"   IC 95%: [{ate_interval_inad[0]:.4f}, {ate_interval_inad[1]:.4f}]")
    
    # 8. Visualizações
    print("\n📊 Gerando Visualizações...")
    criar_visualizacoes_completas(df, analise.modelos, X, feature_names)
    
    # 9. Simulação de Cenários
    print("\n🎯 Simulação de Cenários de Taxa")
    print("-" * 50)
    
    taxas_cenarios = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
    baseline_features = np.median(X, axis=0).reshape(1, -1)
    
    # Replicar features para cada cenário
    X_cenarios = np.tile(baseline_features, (len(taxas_cenarios), 1))
    
    # Predizer efeitos para diferentes taxas
    efeitos_cenarios = cf_dml.effect(X_cenarios)
    
    cenarios_df = pd.DataFrame({
        'Taxa (%)': taxas_cenarios.flatten(),
        'Efeito na Aprovação': efeitos_cenarios.flatten(),
        'Prob. Aprovação Estimada': np.mean(Y_aprov) + efeitos_cenarios.flatten()
    })
    
    print(cenarios_df.to_string(index=False))
    
    # 10. Resumo Executivo
    print("\n" + "=" * 60)
    print("RESUMO EXECUTIVO")
    print("=" * 60)
    
    print(f"""
    📈 PRINCIPAIS RESULTADOS:
    
    1. EFEITO CAUSAL DA TAXA DE JUROS:
       • Linear DML: {linear_dml.ate(X):.4f}
       • Causal Forest: {cf_dml.ate(X):.4f}
       • DR Learner: {dr_learner.ate(X):.4f}
       
    2. HETEROGENEIDADE DO EFEITO:
       • Variação entre grupos: {efeitos_score['mean'].max() - efeitos_score['mean'].min():.4f}
       • Principal driver: {importance_df.iloc[0]['Feature']}
       
    3. IMPACTO NA INADIMPLÊNCIA:
       • Efeito: {ate_inad:.4f}
       • Significância: {'Sim' if ate_interval_inad[0] * ate_interval_inad[1] > 0 else 'Não'}
       
    4. RECOMENDAÇÕES:
       • Considerar políticas diferenciadas por score de crédito
       • Monitorar elasticidade em diferentes segmentos
       • Otimizar trade-off entre volume e risco
    """)
    
    return df, analise.modelos

# ==========================================
# 6. FUNÇÕES AUXILIARES AVANÇADAS
# ==========================================

def analise_sensibilidade_confounding(modelo, X, Y, T, W, gamma_values=[0.8, 1.0, 1.2, 1.5]):
    """
    Análise de sensibilidade para confounders não observados
    """
    print("\n🔬 Análise de Sensibilidade para Confounding")
    print("-" * 50)
    
    base_ate = modelo.ate(X)
    
    resultados = []
    for gamma in gamma_values:
        # Simular confounding multiplicando o efeito
        adjusted_ate = base_ate * gamma
        
        resultados.append({
            'Gamma': gamma,
            'ATE Ajustado': adjusted_ate,
            'Mudança (%)': (adjusted_ate/base_ate - 1) * 100
        })
    
    sensitivity_df = pd.DataFrame(resultados)
    print(sensitivity_df.to_string(index=False))
    
    return sensitivity_df

def validacao_cruzada_temporal(df, n_splits=5):
    """
    Validação cruzada temporal para avaliar estabilidade
    """
    print("\n⏱️ Validação Cruzada Temporal")
    print("-" * 50)
    
    # Simular índice temporal
    df['time_index'] = np.arange(len(df))
    
    # Criar splits temporais
    split_size = len(df) // n_splits
    
    ates_temporais = []
    
    for i in range(n_splits - 1):
        train_mask = df['time_index'] < (i + 1) * split_size
        test_mask = (df['time_index'] >= (i + 1) * split_size) & \
                   (df['time_index'] < (i + 2) * split_size)
        
        # Preparar dados de treino
        df_train = df[train_mask]
        gerador = DadosFinanciamentoImobiliario()
        Y_train, T_train, X_train, W_train, _, _ = gerador.preparar_dados_dml(
            df_train, 'prob_aprovacao'
        )
        
        # Treinar modelo
        modelo_temporal = LinearDML(
            model_y=RandomForestRegressor(n_estimators=50),
            model_t=RandomForestRegressor(n_estimators=50),
            cv=3
        )
        modelo_temporal.fit(Y_train, T_train, X=X_train, W=W_train)
        
        # Avaliar no teste
        df_test = df[test_mask]
        Y_test, T_test, X_test, W_test, _, _ = gerador.preparar_dados_dml(
            df_test, 'prob_aprovacao'
        )
        
        ate_test = modelo_temporal.ate(X_test)
        ates_temporais.append(ate_test)
        
        print(f"   Período {i+1}: ATE = {ate_test:.4f}")
    
    print(f"\n   Média dos ATEs: {np.mean(ates_temporais):.4f}")
    print(f"   Desvio Padrão: {np.std(ates_temporais):.4f}")
    
    return ates_temporais

# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

if __name__ == "__main__":
    # Executar análise completa
    df_resultados, modelos_treinados = executar_analise_econml()
    
    # Análises adicionais
    if 'causal_forest' in modelos_treinados:
        # Preparar dados para análise adicional
        gerador = DadosFinanciamentoImobiliario()
        Y, T, X, W, _, _ = gerador.preparar_dados_dml(df_resultados, 'prob_aprovacao')
        
        # Análise de sensibilidade
        analise_sensibilidade_confounding(
            modelos_treinados['causal_forest'], 
            X, Y, T, W
        )
        
        # Validação temporal
        ates_temporais = validacao_cruzada_temporal(df_resultados)
    
    print("\n✅ Análise Concluída com Sucesso!")
    print("=" * 60)
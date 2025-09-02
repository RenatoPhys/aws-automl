# Exemplos de Uso - Feature Query Generator Autossuficiente
# =========================================================

import pandas as pd
import awswrangler as wr
from datetime import datetime

# -----------------------------------------------------------------------------
# EXEMPLO 1: Query Autossuficiente Básica
# -----------------------------------------------------------------------------

from feature_query_generator import FeatureQueryGenerator

# Configuração básica
generator = FeatureQueryGenerator(
    base_name="analytics.tb_transacoes_cartao",
    partition_column="anomes",
    cpf_column="nu_cpf",
    aggregations=['sum', 'avg', 'max', 'min', 'count', 'std']
)

# Variáveis para feature engineering
numeric_vars = [
    'vl_transacao',
    'qt_parcelas', 
    'vl_fatura',
    'vl_limite_utilizado'
]

categorical_vars = [
    'tp_transacao',
    'cd_mcc',
    'nm_estabelecimento',
    'fl_online'
]

# Gera query com data de referência específica
query_basica = generator.generate_optimized_query(
    numeric_vars=numeric_vars,
    categorical_vars=categorical_vars,
    time_breaks=[1, 3, 6, 12],  # Últimos 1, 3, 6 e 12 meses
    reference_date='202412'  # Data de referência explícita
)

print("="*80)
print("QUERY BÁSICA - COM DATA DE REFERÊNCIA ESPECÍFICA")
print("="*80)
print(query_basica)
print("\n")

# -----------------------------------------------------------------------------
# EXEMPLO 2: Query com Filtros Adicionais
# -----------------------------------------------------------------------------

# Adiciona filtros específicos do negócio
query_com_filtros = generator.generate_optimized_query(
    numeric_vars=numeric_vars,
    categorical_vars=categorical_vars,
    time_breaks=[3, 6, 12],
    additional_filters="vl_transacao > 0 AND tp_transacao IN ('CREDITO', 'DEBITO')"
)

print("="*80)
print("QUERY COM FILTROS ADICIONAIS")
print("="*80)
print(query_com_filtros[:1000] + "...")  # Mostra apenas parte da query
print("\n")

# -----------------------------------------------------------------------------
# EXEMPLO 3: Query com Razões (Ratio Features)
# -----------------------------------------------------------------------------

# Gera features de razão entre variáveis
ratio_features = generator.generate_ratio_features(
    numerator_vars=['valor_transacao', 'valor_fatura'],
    denominator_vars=['vl_limite_utilizado', 'qt_parcelas'],
    reference_date='202412',
    time_breaks=[1, 3, 6]
)

print("="*80)
print("FEATURES DE RAZÃO (RATIO)")
print("="*80)
for feature in ratio_features[:3]:  # Mostra apenas algumas
    print(feature)
print("\n")

# -----------------------------------------------------------------------------
# EXEMPLO 4: Query de Validação e Identificação de Partição Mais Recente
# -----------------------------------------------------------------------------

# Query para validar a qualidade dos dados
query_validacao = generator.generate_validation_query(reference_date='202412')

print("="*80)
print("QUERY DE VALIDAÇÃO DE DADOS")
print("="*80)
print(query_validacao)
print("\n")

# Query para obter a partição mais recente
query_ultima_particao = generator.get_latest_partition_query()

print("="*80)
print("QUERY PARA OBTER ÚLTIMA PARTIÇÃO")
print("="*80)
print(query_ultima_particao)
print("\n")

# Como usar para processar sempre a última partição disponível:
"""
# 1. Primeiro descobre qual é a última partição
df_ultima = wr.athena.read_sql_query(
    sql=query_ultima_particao,
    database=database
)
ultima_particao = df_ultima['ultima_particao'].iloc[0]

# 2. Usa essa partição para gerar as features
query_features = generator.generate_optimized_query(
    numeric_vars=numeric_vars,
    categorical_vars=categorical_vars,
    reference_date=ultima_particao
)
"""

# -----------------------------------------------------------------------------
# EXEMPLO 5: Pipeline Completo com Execução no Athena
# -----------------------------------------------------------------------------

def criar_book_features(
    database: str,
    s3_output: str,
    reference_date: str = None,
    usar_ultima_particao: bool = False
):
    """
    Pipeline completo para criação de book de features.
    
    Args:
        database: Nome do database no Athena
        s3_output: Caminho S3 para output
        reference_date: Data de referência (YYYYMM). Se None, usa mês anterior
        usar_ultima_particao: Se True, ignora reference_date e usa a última partição disponível
    """
    
    # Configura o gerador
    generator = FeatureQueryGenerator(
        base_name="analytics.tb_transacoes_consolidadas",
        partition_column="anomes",
        cpf_column="cpf"
    )
    
    # Determina qual partição processar
    if usar_ultima_particao:
        print("Identificando última partição disponível...")
        query_ultima = generator.get_latest_partition_query()
        df_ultima = wr.athena.read_sql_query(
            sql=query_ultima,
            database=database,
            s3_output=s3_output
        )
        reference_date = df_ultima['ultima_particao'].iloc[0]
        print(f"Última partição encontrada: {reference_date}")
    elif not reference_date:
        # Se não especificar data, usa o mês anterior
        today = datetime.now()
        last_month = today.replace(day=1) - timedelta(days=1)
        reference_date = last_month.strftime('%Y%m')
    
    print(f"Processando features para a data de referência: {reference_date}")
    
    # Define variáveis
    numeric_vars = [
        'valor_total',
        'quantidade_transacoes',
        'ticket_medio',
        'dias_desde_ultima_compra'
    ]
    
    categorical_vars = [
        'categoria_principal',
        'canal_preferencial',
        'tipo_pagamento'
    ]
    
    # 1. Primeiro valida os dados
    print("Etapa 1: Validando dados...")
    query_val = generator.generate_validation_query(reference_date=reference_date)
    df_validacao = wr.athena.read_sql_query(
        sql=query_val,
        database=database,
        s3_output=s3_output
    )
    
    print(f"Validação:")
    print(df_validacao)
    
    # 2. Gera as features
    print("\nEtapa 2: Gerando features...")
    query_features = generator.generate_optimized_query(
        numeric_vars=numeric_vars,
        categorical_vars=categorical_vars,
        time_breaks=[1, 3, 6, 12, 24],
        reference_date=reference_date
    )
    
    df_features = wr.athena.read_sql_query(
        sql=query_features,
        database=database,
        s3_output=s3_output,
        keep_files=False,
        ctas_approach=True  # Usa CTAS para melhor performance
    )
    
    print(f"Features geradas: {df_features.shape}")
    print(f"Data de referência processada: {reference_date}")
    
    # 3. Validações pós-processamento
    print("\nEtapa 3: Validações pós-processamento...")
    
    # Verifica duplicados
    duplicados = df_features[df_features.duplicated(subset=['cpf'], keep=False)]
    if len(duplicados) > 0:
        print(f"⚠️  ALERTA: {len(duplicados)} CPFs duplicados encontrados!")
    else:
        print("✅ OK: Sem CPFs duplicados")
    
    # Verifica nulos
    pct_nulos = (df_features.isnull().sum() / len(df_features) * 100)
    colunas_problema = pct_nulos[pct_nulos > 50]
    if len(colunas_problema) > 0:
        print(f"⚠️  ALERTA: {len(colunas_problema)} colunas com >50% nulos")
    else:
        print("✅ OK: Nenhuma coluna com excesso de nulos")
    
    # 4. Salva o resultado
    if s3_output:
        output_path = f"s3://seu-bucket/feature-store/book_visao_cpf/dt_referencia={reference_date}/"
        print(f"\nEtapa 4: Salvando em {output_path}...")
        wr.s3.to_parquet(
            df=df_features,
            path=output_path,
            mode='overwrite'
        )
        print("✅ Features salvas com sucesso!")
    
    return df_features

# Exemplo de execução (descomente para rodar)
"""
# Opção 1: Especificando a data de referência
df_book = criar_book_features(
    database="analytics_prod",
    s3_output="s3://seu-bucket/athena-results/",
    reference_date="202412"
)

# Opção 2: Usando sempre a última partição disponível
df_book = criar_book_features(
    database="analytics_prod",
    s3_output="s3://seu-bucket/athena-results/",
    usar_ultima_particao=True
)
"""

# -----------------------------------------------------------------------------
# EXEMPLO 6: Query com Quebras Categóricas
# -----------------------------------------------------------------------------

generator2 = FeatureQueryGenerator(
    base_name="analytics.tb_vendas",
    partition_column="anomes",
    cpf_column="cpf_cliente"
)

# Query com quebras categóricas
query_categoricas = generator2.generate_query_with_categorical_breaks(
    numeric_vars=['valor_venda', 'quantidade_itens'],
    categorical_breaks=['segmento', 'regiao', 'canal_venda'],
    time_breaks=[1, 3, 6],
    reference_date='202412'
)

print("="*80)
print("QUERY COM QUEBRAS CATEGÓRICAS")
print("="*80)
print(query_categoricas[:1000] + "...")
print("\n")

# -----------------------------------------------------------------------------
# EXEMPLO 7: Classe Estendida com Features Específicas do Negócio
# -----------------------------------------------------------------------------

# Assumindo que FeatureQueryGenerator foi importado
# from feature_query_generator import FeatureQueryGenerator

class FeatureGeneratorCredito(FeatureQueryGenerator):
    """
    Extensão específica para features de crédito.
    """
    
    def generate_risk_features_query(
        self,
        reference_date: str,
        time_breaks: List[int] = None
    ) -> str:
        """
        Gera features específicas para modelos de risco de crédito.
        """
        time_breaks = time_breaks or [1, 3, 6, 12]
        max_months = max(time_breaks)
        min_start_date = self._calculate_start_date(reference_date, max_months)
        
        # Features específicas de risco
        risk_features = []
        
        for months in time_breaks:
            start_date = self._calculate_start_date(reference_date, months)
            
            # Taxa de utilização do limite
            risk_features.append(
                f"AVG(CASE WHEN {self.partition_column} >= '{start_date}' "
                f"AND {self.partition_column} <= '{reference_date}' "
                f"THEN vl_utilizado / NULLIF(vl_limite, 0) END) AS taxa_utilizacao_{months}m"
            )
            
            # Frequência de atrasos
            risk_features.append(
                f"SUM(CASE WHEN {self.partition_column} >= '{start_date}' "
                f"AND {self.partition_column} <= '{reference_date}' "
                f"AND fl_atraso = 1 THEN 1 ELSE 0 END) AS freq_atrasos_{months}m"
            )
            
            # Valor máximo de atraso
            risk_features.append(
                f"MAX(CASE WHEN {self.partition_column} >= '{start_date}' "
                f"AND {self.partition_column} <= '{reference_date}' "
                f"THEN dias_atraso END) AS max_dias_atraso_{months}m"
            )
            
            # Variabilidade de pagamentos
            risk_features.append(
                f"STDDEV(CASE WHEN {self.partition_column} >= '{start_date}' "
                f"AND {self.partition_column} <= '{reference_date}' "
                f"THEN vl_pagamento END) AS std_pagamentos_{months}m"
            )
        
        query = f"""
SELECT 
    {self.cpf_column},
    '{reference_date}' AS dt_referencia,
    CURRENT_TIMESTAMP AS dt_processamento,
    -- Features de risco
    {',\n    '.join(risk_features)},
    -- Score composto de risco
    (
        COALESCE(AVG(CASE WHEN fl_atraso = 1 THEN 1 ELSE 0 END), 0) * 0.4 +
        COALESCE(AVG(vl_utilizado / NULLIF(vl_limite, 0)), 0) * 0.3 +
        COALESCE(MAX(dias_atraso) / 90.0, 0) * 0.3
    ) AS score_risco_composto
FROM {self.base_name}
WHERE {self.partition_column} >= '{min_start_date}'
    AND {self.partition_column} <= '{reference_date}'
GROUP BY {self.cpf_column}
"""
        return query

# Uso da classe estendida
generator_credito = FeatureGeneratorCredito(
    base_name="analytics.tb_cartao_credito",
    partition_column="anomes",
    cpf_column="cpf"
)

query_risco = generator_credito.generate_risk_features_query(
    reference_date='202412',
    time_breaks=[1, 3, 6, 12]
)
print("="*80)
print("QUERY DE FEATURES DE RISCO DE CRÉDITO")
print("="*80)
print(query_risco[:800] + "...")
print("\n")

# -----------------------------------------------------------------------------
# EXEMPLO 8: Configurações para Diferentes Tipos de Modelos
# -----------------------------------------------------------------------------

# Dicionário de configurações por tipo de modelo
CONFIGS_MODELOS = {
    'churn': {
        'numeric_vars': [
            'dias_inatividade',
            'reducao_uso_percentual',
            'quantidade_reclamacoes',
            'nps_score',
            'valor_lifetime'
        ],
        'categorical_vars': [
            'ultimo_canal_contato',
            'motivo_ultimo_contato',
            'segmento_valor'
        ],
        'time_breaks': [1, 3, 6],
        'aggregations': ['avg', 'max', 'min', 'std']
    },
    'cross_sell': {
        'numeric_vars': [
            'qtd_produtos_ativos',
            'valor_total_produtos',
            'tempo_relacionamento_meses',
            'score_satisfacao'
        ],
        'categorical_vars': [
            'produtos_possui',
            'canal_aquisicao',
            'perfil_investidor'
        ],
        'time_breaks': [3, 6, 12],
        'aggregations': ['sum', 'avg', 'max', 'count']
    },
    'fraude': {
        'numeric_vars': [
            'valor_transacao',
            'frequencia_transacao',
            'distancia_media_transacoes',
            'hora_transacao'
        ],
        'categorical_vars': [
            'pais_transacao',
            'tipo_estabelecimento',
            'tipo_cartao'
        ],
        'time_breaks': [1, 7, 30],  # Em dias para fraude
        'aggregations': ['sum', 'avg', 'max', 'std', 'count']
    }
}

# Função para gerar query baseada no tipo de modelo
def gerar_query_por_modelo(tipo_modelo: str, base_name: str, reference_date: str):
    """
    Gera query específica para cada tipo de modelo.
    """
    if tipo_modelo not in CONFIGS_MODELOS:
        raise ValueError(f"Tipo de modelo '{tipo_modelo}' não configurado")
    
    config = CONFIGS_MODELOS[tipo_modelo]
    
    generator = FeatureQueryGenerator(
        base_name=base_name,
        aggregations=config['aggregations']
    )
    
    return generator.generate_optimized_query(
        numeric_vars=config['numeric_vars'],
        categorical_vars=config['categorical_vars'],
        time_breaks=config['time_breaks'],
        reference_date=reference_date
    )

# Exemplo de uso
query_churn = gerar_query_por_modelo(
    'churn', 
    'analytics.tb_comportamento_cliente',
    '202412'
)
print("="*80)
print(f"QUERY PARA MODELO DE CHURN")
print("="*80)
print(query_churn[:600] + "...")
print("\n")

print("✅ Todos os exemplos de queries foram gerados com sucesso!")
print("📝 As queries recebem a data de referência como parâmetro explícito.")
print("🚀 Prontas para execução no AWS Athena via awswrangler!")
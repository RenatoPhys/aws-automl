# Inicializa o gerador
generator = FeatureQueryGenerator(
    base_name="tb_transacoes",
    partition_column="anomes",
    cpf_column="cpf",
    aggregations= ['sum']
)

# =========================================
# EXEMPLO 1: Usando ratio_vars como dicionário
# =========================================

# Definindo razões com dicionário
ratio_vars = {
    'valor_pago': 'valor_fatura',                    # Razão simples: um numerador, um denominador
    'qtd_transacoes': ['qtd_dias', 'qtd_lojas'],    # Razão múltipla: um numerador, múltiplos denominadores
    #'valor_total': ['limite_cartao', 'renda']        # Outro exemplo com múltiplos denominadores
}

# =========================================
# EXEMPLO 2: Usando categorical_counts
# =========================================

# Definindo contagens categóricas específicas
categorical_counts = {
    #'tipo_transacao': ['DEBITO', 'CREDITO'],
    'canal': ['ONLINE', 'LOJA FÍSICA'],
    #'categoria_mcc': ['SUPERMERCADO','RESTAURANTE'],
    #'flag_internacional': ['S', 'N']
}

# =========================================
# EXEMPLO 3: Query completa com todas as features
# =========================================

query_completa = generator.generate_optimized_query(
    numeric_vars=['valor_transacao'],
    categorical_vars=['tipo_estabelecimento'],  # Para nunique e count geral
    time_breaks=[1, 3],
    reference_date='202412',
    ratio_vars=ratio_vars,                      # NOVO: usando dicionário
    categorical_counts=categorical_counts,       # NOVO: contagens específicas
    additional_filters="valor_transacao > 0",
    
)

print("=" * 80)
print("QUERY COMPLETA COM NOVOS RECURSOS")
print("=" * 80)
print(query_completa)



# Exemplos Práticos - Novas Funcionalidades do Feature Query Generator
# =====================================================================

import pandas as pd
import awswrangler as wr
from datetime import datetime, timedelta
from typing import List, Dict, Union
from feature_query_generator import FeatureQueryGenerator

# =============================================================================
# EXEMPLO 1: VARIÁVEIS DE RAZÃO COM DICIONÁRIO
# =============================================================================

def exemplo_razoes_credito():
    """
    Exemplo de uso de variáveis de razão para análise de crédito.
    """
    generator = FeatureQueryGenerator(
        base_name="analytics.tb_cartao_credito",
        partition_column="anomes",
        cpf_column="cpf"
    )
    
    # Definindo razões relevantes para análise de crédito
    # Formato: {numerador: denominador(es)}
    ratio_vars = {
        # Taxa de utilização do cartão
        'valor_utilizado': 'limite_total',
        
        # Taxa de pagamento
        'valor_pago': ['valor_fatura', 'valor_minimo'],
        
        # Indicadores de comportamento
        'qtd_transacoes': ['qtd_dias_mes', 'qtd_estabelecimentos'],
        
        # Ticket médio
        'valor_total_compras': 'qtd_transacoes',
        
        # Taxa de parcelamento
        'valor_parcelado': 'valor_total_compras',
        
        # Indicador de concentração
        'maior_compra': 'valor_total_compras'
    }
    
    query = generator.generate_optimized_query(
        numeric_vars=['valor_fatura', 'dias_atraso'],
        reference_date='202412',
        time_breaks=[1, 3, 6],
        ratio_vars=ratio_vars
    )
    
    print("=" * 80)
    print("EXEMPLO 1: RAZÕES PARA ANÁLISE DE CRÉDITO")
    print("=" * 80)
    print(query[:1500])
    print("\n✅ Query gerada com razões de crédito!\n")
    
    return query

# =============================================================================
# EXEMPLO 2: CONTAGENS CATEGÓRICAS PARA DETECÇÃO DE FRAUDE
# =============================================================================

def exemplo_contagens_fraude():
    """
    Exemplo de contagens categóricas para modelo de detecção de fraude.
    Demonstra o tratamento de acentos e caracteres especiais.
    """
    generator = FeatureQueryGenerator(
        base_name="analytics.tb_transacoes_online",
        partition_column="data_hora",
        cpf_column="cpf_titular"
    )
    
    # Definindo categorias suspeitas para contar - com acentos e caracteres especiais
    categorical_counts = {
        # Tipos de transação
        'tipo_transacao': [
            'SAQUE_ATM',
            'COMPRA_INTERNACIONAL',
            'TRANSFERÊNCIA_TED',
            'PIX_DESCONHECIDO',
            'DÉBITO_AUTOMÁTICO'
        ],
        
        # Países de alto risco
        'pais_transacao': [
            'NIGÉRIA',
            'RÚSSIA',
            'CHINA',
            'INDONÉSIA',
            'REPÚBLICA_CHECA'
        ],
        
        # Horários suspeitos
        'faixa_horaria': [
            'MADRUGADA_0-6H',
            'NOITE_22-24H',
            'HORÁRIO_COMERCIAL',
            'FIM_DE_SEMANA'
        ],
        
        # MCCs de risco
        'categoria_mcc': [
            'CRIPTO-MOEDA',
            'JOGOS_DE_AZAR',
            'CASH_ADVANCE',
            'MONEY-TRANSFER',
            'LOJAS_DE_CONVENIÊNCIA'
        ],
        
        # Dispositivos
        'tipo_dispositivo': [
            'EMULADOR',
            'ROOTED/JAILBREAK',
            'DESCONHECIDO',
            'MÁQUINA_VIRTUAL'
        ]
    }
    
    query = generator.generate_categorical_count_features(
        categorical_counts=categorical_counts,
        reference_date='202412',
        time_breaks=[1, 7, 30],  # Em dias para fraude
        additional_filters="status_transacao = 'APROVADA'"
    )
    
    print("=" * 80)
    print("EXEMPLO 2: CONTAGENS PARA DETECÇÃO DE FRAUDE (com acentos)")
    print("=" * 80)
    print("Demonstração de normalização de nomes:")
    print("- 'TRANSFERÊNCIA_TED' → tipo_transacao_transferencia_ted_count_1m")
    print("- 'DÉBITO_AUTOMÁTICO' → tipo_transacao_debito_automatico_count_1m")
    print("- 'NIGÉRIA' → pais_transacao_nigeria_count_1m")
    print("- 'RÚSSIA' → pais_transacao_russia_count_1m")
    print("- 'REPÚBLICA_CHECA' → pais_transacao_republica_checa_count_1m")
    print("- 'MADRUGADA_0-6H' → faixa_horaria_madrugada_0_6h_count_1m")
    print("- 'CRIPTO-MOEDA' → categoria_mcc_cripto_moeda_count_1m")
    print("- 'LOJAS_DE_CONVENIÊNCIA' → categoria_mcc_lojas_de_conveniencia_count_1m")
    print("- 'ROOTED/JAILBREAK' → tipo_dispositivo_rooted_jailbreak_count_1m")
    print("- 'MÁQUINA_VIRTUAL' → tipo_dispositivo_maquina_virtual_count_1m")
    print("\n" + "=" * 80)
    print(query[:1500])
    print("\n✅ Query com contagens de categorias de risco e acentos normalizados!\n")
    
    return query

# =============================================================================
# EXEMPLO 3: COMBINANDO RAZÕES E CONTAGENS PARA MODELO DE CHURN
# =============================================================================

def exemplo_completo_churn():
    """
    Exemplo combinando razões e contagens para modelo de churn.
    """
    generator = FeatureQueryGenerator(
        base_name="analytics.tb_comportamento_cliente",
        partition_column="anomes",
        cpf_column="cpf_cliente"
    )
    
    # Variáveis numéricas base
    numeric_vars = [
        'valor_compras',
        'qtd_produtos',
        'valor_desconto',
        'qtd_reclamacoes'
    ]
    
    # Razões indicativas de engajamento
    ratio_vars = {
        # Taxa de uso de desconto
        'valor_desconto': 'valor_compras',
        
        # Frequência de compra
        'qtd_transacoes': 'qtd_meses_ativo',
        
        # Reclamações por compra
        'qtd_reclamacoes': 'qtd_transacoes',
        
        # Diversificação de categorias
        'qtd_categorias_distintas': 'qtd_produtos',
        
        # Taxa de devolução
        'valor_devolvido': 'valor_compras'
    }
    
    # Contagens de comportamentos específicos
    categorical_counts = {
        'canal_compra': [
            'APP_MOBILE',
            'WEBSITE',
            'LOJA_FISICA'
        ],
        
        'tipo_pagamento': [
            'CREDITO_PARCELADO',
            'DEBITO',
            'BOLETO'
        ],
        
        'categoria_nps': [
            'PROMOTOR',
            'NEUTRO',
            'DETRATOR'
        ],
        
        'status_programa_fidelidade': [
            'ATIVO',
            'INATIVO',
            'RESGATOU_PONTOS'
        ],
        
        'tipo_contato': [
            'RECLAMACAO',
            'CANCELAMENTO_TENTATIVA',
            'UPGRADE_PLANO'
        ]
    }
    
    query = generator.generate_optimized_query(
        numeric_vars=numeric_vars,
        reference_date='202412',
        time_breaks=[1, 3, 6, 12],
        ratio_vars=ratio_vars,
        categorical_counts=categorical_counts,
        additional_filters="status_cliente = 'ATIVO'"
    )
    
    print("=" * 80)
    print("EXEMPLO 3: MODELO COMPLETO DE CHURN")
    print("=" * 80)
    print(query[:2000])
    print("\n✅ Query completa para modelo de churn!\n")
    
    return query

# =============================================================================
# EXEMPLO 4: SEGMENTAÇÃO DE CLIENTES COM MÚLTIPLAS DIMENSÕES
# =============================================================================

def exemplo_segmentacao_clientes():
    """
    Exemplo para segmentação de clientes usando contagens categóricas.
    """
    generator = FeatureQueryGenerator(
        base_name="analytics.tb_cliente_360",
        partition_column="anomes",
        cpf_column="cpf"
    )
    
    # Contagens para perfil demográfico e comportamental
    categorical_counts = {
        # Perfil de consumo
        'faixa_horaria_compra': [
            'MANHA_6_12',
            'TARDE_12_18',
            'NOITE_18_24',
            'MADRUGADA_0_6'
        ],
        
        # Preferências de categoria
        'categoria_produto': [
            'ELETRONICOS',
            'MODA_FEMININA',
            'MODA_MASCULINA',
            'CASA_DECORACAO',
            'ESPORTE_LAZER',
            'LIVROS_MIDIA',
            'ALIMENTOS_BEBIDAS'
        ],
        
        # Comportamento promocional
        'resposta_campanha': [
            'ABRIU_EMAIL',
            'CLICOU_LINK',
            'CONVERTEU',
            'NAO_ABRIU'
        ],
        
        # Tipo de dispositivo
        'dispositivo_preferencial': [
            'iOS',
            'ANDROID',
            'DESKTOP',
            'TABLET'
        ]
    }
    
    # Query apenas com contagens para clustering
    query = generator.generate_categorical_count_features(
        categorical_counts=categorical_counts,
        reference_date='202412',
        time_breaks=[3, 6, 12]
    )
    
    print("=" * 80)
    print("EXEMPLO 4: FEATURES PARA SEGMENTAÇÃO")
    print("=" * 80)
    print(query[:1500])
    print("\n✅ Features de contagem para segmentação!\n")
    
    return query

# =============================================================================
# EXEMPLO 5: ANÁLISE DE RISCO COM RATIOS COMPLEXOS
# =============================================================================

def exemplo_analise_risco():
    """
    Exemplo de análise de risco usando múltiplas razões por numerador.
    """
    generator = FeatureQueryGenerator(
        base_name="analytics.tb_credito_consolidado",
        partition_column="anomes",
        cpf_column="cpf"
    )
    
    # Razões complexas para análise de risco
    # Um numerador pode ter múltiplos denominadores
    ratio_vars = {
        # Valor em atraso comparado com múltiplas bases
        'valor_atraso': [
            'valor_fatura_total',
            'limite_credito',
            'renda_declarada'
        ],
        
        # Utilização comparada com diferentes limites
        'valor_utilizado': [
            'limite_total',
            'limite_disponivel',
            'limite_emergencial'
        ],
        
        # Pagamentos comparados com diferentes obrigações
        'valor_pago_mes': [
            'valor_minimo',
            'valor_fatura',
            'valor_total_devido'
        ],
        
        # Indicadores de stress financeiro
        'qtd_produtos_atraso': [
            'qtd_produtos_total',
            'qtd_produtos_ativos'
        ],
        
        # Concentração de dívida
        'maior_divida': [
            'soma_todas_dividas',
            'renda_mensal'
        ]
    }
    
    query = generator.generate_optimized_query(
        numeric_vars=['score_credito', 'meses_relacionamento'],
        reference_date='202412',
        time_breaks=[1, 3, 6],
        ratio_vars=ratio_vars
    )
    
    print("=" * 80)
    print("EXEMPLO 5: ANÁLISE DE RISCO COMPLEXA")
    print("=" * 80)
    print(query[:1500])
    print("\n✅ Query com múltiplas razões por variável!\n")
    
    return query

# =============================================================================
# EXEMPLO 6: PIPELINE COMPLETO COM AS NOVAS FUNCIONALIDADES
# =============================================================================

def pipeline_completo_ml(
    database: str,
    s3_output: str,
    reference_date: str = '202412',
    modelo_tipo: str = 'credito'
):
    """
    Pipeline completo usando as novas funcionalidades.
    
    Args:
        database: Database no Athena
        s3_output: Path S3 para output
        reference_date: Data de referência
        modelo_tipo: Tipo do modelo ('credito', 'fraude', 'churn')
    """
    
    # Configurações por tipo de modelo
    configs = {
        'credito': {
            'base_name': 'analytics.tb_credito',
            'numeric_vars': ['valor_fatura', 'valor_pago', 'limite_utilizado'],
            'ratio_vars': {
                'valor_pago': 'valor_fatura',
                'valor_utilizado': 'limite_total',
                'qtd_atrasos': 'qtd_faturas'
            },
            'categorical_counts': {
                'status_pagamento': ['PAGO', 'ATRASADO', 'MINIMO'],
                'tipo_cliente': ['PRIME', 'STANDARD']
            },
            'time_breaks': [1, 3, 6]
        },
        'fraude': {
            'base_name': 'analytics.tb_transacoes',
            'numeric_vars': ['valor_transacao', 'qtd_tentativas'],
            'ratio_vars': {
                'valor_transacao': 'media_historica',
                'qtd_tentativas_falha': 'qtd_tentativas_total'
            },
            'categorical_counts': {
                'tipo_erro': ['SENHA_INVALIDA', 'CARTAO_BLOQUEADO', 'LIMITE_EXCEDIDO'],
                'origem': ['APP', 'WEB', 'ATM']
            },
            'time_breaks': [1, 7, 30]
        },
        'churn': {
            'base_name': 'analytics.tb_cliente',
            'numeric_vars': ['valor_compras', 'qtd_produtos'],
            'ratio_vars': {
                'qtd_reclamacoes': 'qtd_compras',
                'valor_desconto': 'valor_total'
            },
            'categorical_counts': {
                'tipo_interacao': ['COMPRA', 'RECLAMACAO', 'CANCELAMENTO_TENTATIVA'],
                'canal': ['ONLINE', 'LOJA', 'TELEFONE']
            },
            'time_breaks': [1, 3, 6, 12]
        }
    }
    
    config = configs[modelo_tipo]
    
    # Inicializa gerador
    generator = FeatureQueryGenerator(
        base_name=config['base_name'],
        partition_column='anomes',
        cpf_column='cpf'
    )
    
    # Gera query completa
    query = generator.generate_optimized_query(
        numeric_vars=config['numeric_vars'],
        reference_date=reference_date,
        time_breaks=config['time_breaks'],
        ratio_vars=config['ratio_vars'],
        categorical_counts=config['categorical_counts']
    )
    
    print(f"Executando pipeline para modelo: {modelo_tipo}")
    print(f"Data de referência: {reference_date}")
    print("-" * 50)
    
    # Executa no Athena (descomente para rodar)
    """
    df_features = wr.athena.read_sql_query(
        sql=query,
        database=database,
        s3_output=s3_output,
        ctas_approach=True
    )
    
    # Validações
    print(f"Shape do dataset: {df_features.shape}")
    print(f"Memória utilizada: {df_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Tipos de dados:\n{df_features.dtypes.value_counts()}")
    
    # Salva resultado
    output_path = f"s3://feature-store/{modelo_tipo}/dt={reference_date}/"
    wr.s3.to_parquet(
        df=df_features,
        path=output_path,
        mode='overwrite',
        dataset=True,
        partition_cols=['dt_referencia']
    )
    
    return df_features
    """
    
    return query

# =============================================================================
# EXECUÇÃO DOS EXEMPLOS
# =============================================================================

if __name__ == "__main__":
    
    print("🚀 DEMONSTRAÇÃO DAS NOVAS FUNCIONALIDADES")
    print("=" * 80)
    print()
    
    # Executa todos os exemplos
    exemplo_razoes_credito()
    exemplo_contagens_fraude()
    exemplo_completo_churn()
    exemplo_segmentacao_clientes()
    exemplo_analise_risco()
    
    # Pipeline para diferentes modelos
    for modelo in ['credito', 'fraude', 'churn']:
        print("=" * 80)
        print(f"PIPELINE PARA MODELO: {modelo.upper()}")
        print("=" * 80)
        query = pipeline_completo_ml(
            database="analytics_prod",
            s3_output="s3://bucket/athena-results/",
            reference_date="202412",
            modelo_tipo=modelo
        )
        print(query[:1000] + "...\n")
    
    print("=" * 80)
    print("✅ TODAS AS DEMONSTRAÇÕES EXECUTADAS COM SUCESSO!")
    print("=" * 80)
    print("\n📊 Resumo das novas funcionalidades:")
    print("1. ratio_vars agora aceita dicionário {numerador: denominador(es)}")
    print("2. categorical_counts cria contagens para valores específicos")
    print("3. Ambos podem ser usados no método generate_optimized_query()")
    print("4. Método dedicado generate_categorical_count_features()")
    print("5. Suporte para múltiplos denominadores por numerador")
    print("6. ✨ NOVO: Normalização automática de acentos e caracteres especiais")
    print("   - Remove acentos: 'DÉBITO' → 'debito'")
    print("   - Substitui espaços: 'LOJA FÍSICA' → 'loja_fisica'")
    print("   - Remove caracteres especiais: 'CRIPTO-MOEDA' → 'cripto_moeda'")
    print("   - Trata barras: 'ROOTED/JAILBREAK' → 'rooted_jailbreak'")
    
    # Demonstração adicional de normalização
    print("\n" + "=" * 80)
    print("DEMONSTRAÇÃO DE NORMALIZAÇÃO DE CARACTERES ESPECIAIS")
    print("=" * 80)
    
    generator_demo = FeatureQueryGenerator(
        base_name="tb_demo",
        partition_column="anomes",
        cpf_column="cpf"
    )
    
    # Teste de normalização com diversos caracteres especiais
    test_cases = {
        'CAFÉ COM AÇÚCAR': 'cafe_com_acucar',
        'SÃO PAULO': 'sao_paulo',
        'PARANÁ': 'parana',
        'AMANHÃ': 'amanha',
        'AÇÃO/REAÇÃO': 'acao_reacao',
        'R$ 100,00': 'r_100_00',
        'E-COMMERCE': 'e_commerce',
        '24/7': '24_7',
        'CARTÃO CRÉDITO': 'cartao_credito',
        'TRANSFERÊNCIA-PIX': 'transferencia_pix'
    }
    
    print("\nExemplos de normalização de valores categóricos:")
    for original, expected in test_cases.items():
        normalized = generator_demo._normalize_string(original)
        status = "✓" if normalized == expected else "✗"
        print(f"{status} '{original}' → '{normalized}'")
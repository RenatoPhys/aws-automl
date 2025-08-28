# -*- coding: utf-8 -*-

import logging
import pandas as pd
import awswrangler as wr
from typing import List, Dict, Any

# --- Configuração do Logging ---
# Em vez de 'print', usamos logging para um controle melhor das mensagens
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configurações Gerais do Projeto ---
# Centralize todas as suas configurações aqui para fácil manutenção
class Config:
    DATABASE_NAME = 'workspace_db'
    WORKGROUP = 'analytics-workgroup'
    S3_OUTPUT_BUCKET = 'seu-bucket-de-projeto-aqui' # <-- IMPORTANTE: Substitua pelo seu bucket
    METADATA_FILE_PATH = 'tabelas_metadata.xlsx'
    BASE_PUBLICO_TARGET = 'nome_da_sua_tabela_publico_target' # <-- IMPORTANTE: Substitua pelo nome da tabela

class FeaturePipeline:
    """
    Orquestra a criação de tabelas de features juntando uma base de público/alvo
    com diversas tabelas explicativas, baseado em um arquivo de metadados.
    """
    def __init__(self, config: Config, safras_para_processar: List[int]):
        """
        Inicializa o pipeline com as configurações necessárias.

        Args:
            config (Config): Objeto com as configurações do projeto.
            safras_para_processar (List[int]): Lista de safras (anomes) a serem processadas.
        """
        self.config = config
        self.safras_para_processar = safras_para_processar
        self.metadata_df = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        """
        Carrega e valida o arquivo Excel de metadados.
        """
        try:
            logging.info(f"Carregando metadados de '{self.config.METADATA_FILE_PATH}'...")
            df = pd.read_excel(self.config.METADATA_FILE_PATH)
            
            # Validação básica para garantir que as colunas necessárias existem
            required_cols = ['Nome', 'nom_doc', 'n_dig_doc', 'nom_safra', 'tipo_safra', 'Defasagem']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"O arquivo de metadados deve conter as colunas: {required_cols}")
            
            logging.info("Metadados carregados com sucesso.")
            return df
        except FileNotFoundError:
            logging.error(f"Erro: Arquivo de metadados '{self.config.METADATA_FILE_PATH}' não encontrado.")
            raise
        except Exception as e:
            logging.error(f"Erro ao carregar ou validar metadados: {e}")
            raise

    def _get_numeric_columns(self, table_name: str) -> List[str]:
        """
        Obtém as colunas numéricas de uma tabela no Athena de forma eficiente.
        """
        try:
            # wr.catalog.get_table_types é mais eficiente que um 'SELECT ... LIMIT'
            table_types = wr.catalog.get_table_types(
                database=self.config.DATABASE_NAME,
                table=table_name
            )
            numeric_types = ['bigint', 'int', 'smallint', 'tinyint', 'double', 'float', 'decimal']
            return [col for col, dtype in table_types.items() if any(t in dtype for t in numeric_types)]
        except Exception as e:
            logging.error(f"Não foi possível obter os tipos de coluna para a tabela {table_name}: {e}")
            return []

    def _calculate_target_anomes(self, anomes: int, defasagem_str: str) -> int:
        """
        Calcula o anomes da tabela de feature com base na defasagem.
        Exemplo: anomes=202305, defasagem='M-1' -> retorna 202304.
        """
        if not isinstance(defasagem_str, str) or '-' not in defasagem_str:
            logging.warning(f"Defasagem '{defasagem_str}' em formato inválido. Usando defasagem 0.")
            return anomes
            
        offset = int(defasagem_str.split('-')[1])
        
        # Usando pandas para tratar a data de forma robusta
        periodo_atual = pd.Period(str(anomes), freq='M')
        periodo_alvo = periodo_atual - pd.offsets.MonthEnd(offset)
        
        return int(periodo_alvo.strftime('%Y%m'))

    def process_feature_table(self, table_metadata: Dict[str, Any]):
        """
        Processa uma única tabela de feature: cria a tabela final e insere as partições.
        """
        feature_table_name = table_metadata['Nome']
        logging.info(f"--- Iniciando processamento para a tabela de features: {feature_table_name} ---")

        # Define o nome da tabela final
        final_table_name_suffix = feature_table_name.split('.')[-1]
        final_table_name = f"{self.config.BASE_PUBLICO_TARGET}_{final_table_name_suffix}"
        s3_path = f"s3://{self.config.S3_OUTPUT_BUCKET}/temp/{final_table_name_suffix}/"

        # 1. Obter colunas numéricas da tabela de feature
        numeric_cols = self._get_numeric_columns(feature_table_name)
        if not numeric_cols:
            logging.warning(f"Nenhuma coluna numérica encontrada para '{feature_table_name}'. Pulando esta tabela.")
            return

        # 2. Construir e executar a query CREATE TABLE
        cols_for_create = ',\n'.join([f"  `{col}` double" for col in numeric_cols])
        
        create_query = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS `{self.config.DATABASE_NAME}`.`{final_table_name}` (
          `cpf_publico_target` bigint,
          `y` int,
        {cols_for_create}
        )
        PARTITIONED BY (`anomes` int)
        STORED AS PARQUET
        LOCATION '{s3_path}'
        """
        
        logging.info(f"Criando tabela final: {final_table_name}")
        wr.athena.read_sql_query(
            sql=create_query,
            database=self.config.DATABASE_NAME,
            workgroup=self.config.WORKGROUP,
            ctas_approach=False
        )
        logging.info(f"Tabela '{final_table_name}' criada ou já existente.")

        # 3. Inserir dados para cada safra (partição)
        for anomes_particao in self.safras_para_processar:
            try:
                logging.info(f"Processando partição anomes={anomes_particao} para {final_table_name}...")
                
                # Parâmetros da tabela de feature
                doc_field_name = table_metadata['nom_doc'].lower()
                doc_num_digits = int(table_metadata['n_dig_doc'])
                anomes_field_name = table_metadata['nom_safra'].lower()
                defasagem = table_metadata['Defasagem']
                
                # Calcula qual anomes buscar na tabela de feature
                anomes_feature_table = self._calculate_target_anomes(anomes_particao, defasagem)

                # Monta a condição de JOIN para o CPF
                join_condition = ""
                if doc_num_digits == 8: # Raiz do CPF
                    join_condition = f"SUBSTR(CAST(a.cpf AS VARCHAR), 1, 8) = CAST(b.{doc_field_name} AS VARCHAR)"
                elif doc_num_digits == 11: # CPF completo
                    join_condition = f"a.cpf = b.{doc_field_name}"
                else:
                    logging.warning(f"Número de dígitos de documento '{doc_num_digits}' não suportado. Pulando partição.")
                    continue

                cols_for_select = ', '.join([f"b.`{col}`" for col in numeric_cols])

                insert_query = f"""
                INSERT INTO `{self.config.DATABASE_NAME}`.`{final_table_name}`
                SELECT
                  a.cpf AS cpf_publico_target,
                  a.y,
                  {cols_for_select},
                  a.anomes
                FROM `{self.config.DATABASE_NAME}`.`{self.config.BASE_PUBLICO_TARGET}` a
                LEFT JOIN `{feature_table_name}` b ON {join_condition} 
                                               AND b.{anomes_field_name} = {anomes_feature_table}
                WHERE a.anomes = {anomes_particao}
                """
                
                # Executa a query de inserção
                wr.athena.read_sql_query(
                    sql=insert_query,
                    database=self.config.DATABASE_NAME,
                    workgroup=self.config.WORKGROUP,
                    ctas_approach=False
                )
                logging.info(f"    Partição anomes={anomes_particao} da tabela {final_table_name} incluída com sucesso.")

            except Exception as e:
                logging.error(f"    Falha ao processar partição anomes={anomes_particao} para {final_table_name}: {e}")
                continue # Continua para a próxima partição

    def run(self):
        """
        Executa o pipeline completo para todas as tabelas listadas nos metadados.
        """
        logging.info("##### INICIANDO PIPELINE DE CRIAÇÃO DE TABELAS FEATURES #####")
        if self.metadata_df is None:
            logging.error("Pipeline não pode ser executado pois os metadados não foram carregados.")
            return
            
        for index, row in self.metadata_df.iterrows():
            try:
                self.process_feature_table(row.to_dict())
            except Exception as e:
                logging.error(f"Ocorreu um erro inesperado ao processar a tabela {row['Nome']}. Pulando para a próxima. Erro: {e}")
        
        logging.info("##### PIPELINE FINALIZADO #####")


# --- Ponto de Entrada do Script ---
if __name__ == '__main__':
    # Defina aqui as safras (anomes) que você quer processar
    SAFRAS_A_PROCESSAR = [202301, 202302, 202303]

    # Instancia e executa o pipeline
    config = Config()
    pipeline = FeaturePipeline(config, safras_para_processar=SAFRAS_A_PROCESSAR)
    pipeline.run()
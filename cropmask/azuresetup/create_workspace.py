import yaml
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication 

with open('../../azure_configs.yaml') as f:
    configs = yaml.safe_load(f)

ws = Workspace.create(name=configs['account']['workspace_name'],
                      subscription_id=configs['account']['subscription_id'], 
                      resource_group=configs['account']['resource_group'],
                      location=configs['account']['location'],
                      auth=ServicePrincipalAuthentication(
                          configs['account']['tenant_id'],
                          configs['account']['app_id'],
                          configs['account']['app_key']
                          )
                     )
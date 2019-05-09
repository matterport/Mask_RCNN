import yaml
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

with open("/home/rave/azure_configs.yaml") as f:
    configs = yaml.safe_load(f)

ws = Workspace.create(
    name=configs["account"]["workspace_name"],
    subscription_id=configs["account"]["subscription_id"],
    resource_group=configs["account"]["perm_resource_group"], # make sure this is different from terraform's resource group or terraform desstroy will delete it, very bad!!!
    location=configs["account"]["location"],
    auth=ServicePrincipalAuthentication(
        configs["account"]["tenant_id"],
        configs["account"]["app_id"],
        configs["account"]["app_key"],
    ),
    storage_account=configs["account"]['resource_id'] # get from properties of storage account
)

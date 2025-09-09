from agents.fql import FQLAgent
from agents.dfql import DFQLAgent
from agents.difql import DIFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.drebrac import DReBRACAgent
from agents.ifql import IFQLAgent
from agents.diql import DIQLAgent
from agents.trigflow import TrigFQLAgent
from agents.dtrigflow import DTrigFQLAgent
from agents.retrigflow import ReTrigFQLAgent

agents = dict(
    iql=IQLAgent,
    diql=DIQLAgent,
    ifql=IFQLAgent,
    difql=DIFQLAgent,
    trigflow = TrigFQLAgent,
    dtrigflow = DTrigFQLAgent,
    fql=FQLAgent,
    rebrac=ReBRACAgent,
    drebrac=DReBRACAgent,
    dfql=DFQLAgent,
    retrigflow=ReTrigFQLAgent,
)

from agents.fql import FQLAgent
from agents.difql import DIFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.drebrac import DReBRACAgent
from agents.ifql import IFQLAgent
from agents.diql import DIQLAgent
from agents.trigflow import TrigFQLAgent
from agents.dtrigflow import DTrigFQLAgent
from agents.retrigflow import ReTrigFQLAgent
from agents.dsfql import DSFQLAgent
from agents.dmfql import DMFQLAgent
from agents.mfql import MFQLAgent
from agents.sfql import SFQLAgent
from agents.trigql import TrigQLAgent
from agents.dtrigql import DTrigQLAgent
from agents.dmfrebrac import DMFReBRACAgent
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
    retrigflow=ReTrigFQLAgent,
    dsfql=DSFQLAgent,
    dmfql=DMFQLAgent,
    mfql=MFQLAgent,
    sfql=SFQLAgent,
    trigql=TrigQLAgent,
    dtrigql = DTrigQLAgent,
    dmfrebrac = DMFReBRACAgent,
)

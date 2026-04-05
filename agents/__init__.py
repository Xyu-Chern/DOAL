from agents.fql import FQLAgent
from agents.difql import DIFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent

from agents.ifql import IFQLAgent
from agents.diql import DIQLAgent
from agents.trigflow import TrigFQLAgent
from agents.dtrigflow import DTrigFQLAgent

from agents.dmfql import DMFQLAgent
from agents.mfql import MFQLAgent

from agents.dmfrebrac import DMFReBRACAgent
from agents.meanflowql import MeanFlowQL_Agent
from agents.fac import FACAgent
from agents.dfr import DFRAgent
from agents.dn import DNAgent

agents = dict(
    iql=IQLAgent,
    diql=DIQLAgent,
    dn=DNAgent,
    ifql=IFQLAgent,
    dfr=DFRAgent,
    difql=DIFQLAgent,
    trigflow=TrigFQLAgent,
    dtrigflow=DTrigFQLAgent,
    fql=FQLAgent,
    fac=FACAgent,
    meanflowql=MeanFlowQL_Agent,
    rebrac=ReBRACAgent,
    dmfql=DMFQLAgent,
    mfql=MFQLAgent,
    dmfrebrac = DMFReBRACAgent,
)

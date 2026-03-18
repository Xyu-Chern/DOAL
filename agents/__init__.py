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
from agents.dmfrebrac_jit import DMFReBRAC_jitAgent
from agents.fac import FACAgent

agents = dict(
    iql=IQLAgent,
    diql=DIQLAgent,
    ifql=IFQLAgent,
    difql=DIFQLAgent,
    trigflow = TrigFQLAgent,
    dtrigflow = DTrigFQLAgent,
    fql=FQLAgent,
    fac=FACAgent,
    dmfrebrac_jit=DMFReBRAC_jitAgent,
    rebrac=ReBRACAgent,
    dmfql=DMFQLAgent,
    mfql=MFQLAgent,
    dmfrebrac = DMFReBRACAgent,
)

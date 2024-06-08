from enum import Enum


class Event(Enum):
    guatemala_volcano = "guatemala-volcano"
    hurricane_florence = "hurricane-florence"
    hurricane_harvey = "hurricane-harvey"
    hurricane_matthew = "hurricane-matthew"
    hurricane_michael = "hurricane-michael"
    joplin_tornado = "joplin-tornado"
    lower_puna_volcano = "lower-puna-volcano"
    mexico_earthquake = "mexico-earthquake"
    midwest_flooding = "midwest-flooding"
    moore_tornado = "moore-tornado"
    nepal_flooding = "nepal-flooding"
    palu_tsunami = "palu-tsunami"
    pinery_bushfire = "pinery-bushfire"
    portugal_wildfire = "portugal-wildfire"
    santa_rosa_wildfire = "santa-rosa-wildfire"
    socal_fire = "socal-fire"
    sunda_tsunami = "sunda-tsunami"
    tuscaloosa_tornado = "tuscaloosa-tornado"
    woolsey_fire = "woolsey-fire"


class _SubsetBase:
    """Base class for dataset subset (challenge split)"""

    events: set[Event]


class Test(_SubsetBase):
    """Events from "Test" challenge split"""

    events = {
        Event.guatemala_volcano,
        Event.hurricane_florence,
        Event.hurricane_harvey,
        Event.hurricane_matthew,
        Event.hurricane_michael,
        Event.mexico_earthquake,
        Event.midwest_flooding,
        Event.palu_tsunami,
        Event.santa_rosa_wildfire,
        Event.socal_fire,
    }


class Hold(_SubsetBase):
    """Events from "Hold" challenge split"""

    events = {
        Event.guatemala_volcano,
        Event.hurricane_florence,
        Event.hurricane_harvey,
        Event.hurricane_matthew,
        Event.hurricane_michael,
        Event.mexico_earthquake,
        Event.midwest_flooding,
        Event.palu_tsunami,
        Event.santa_rosa_wildfire,
        Event.socal_fire,
    }


class Tier1(_SubsetBase):
    """Events from "Tier 1" challenge split"""

    events = {
        Event.guatemala_volcano,
        Event.hurricane_florence,
        Event.hurricane_harvey,
        Event.hurricane_matthew,
        Event.hurricane_michael,
        Event.mexico_earthquake,
        Event.midwest_flooding,
        Event.palu_tsunami,
        Event.santa_rosa_wildfire,
        Event.socal_fire,
    }


class Tier3(_SubsetBase):
    """Events from "Tier 3" challenge split"""

    events = {
        Event.joplin_tornado,
        Event.lower_puna_volcano,
        Event.moore_tornado,
        Event.nepal_flooding,
        Event.pinery_bushfire,
        Event.portugal_wildfire,
        Event.sunda_tsunami,
        Event.tuscaloosa_tornado,
        Event.woolsey_fire,
    }


Subset = type[Test | Hold | Tier1 | Tier3]

from ignite.engine import EventEnum


class EvaluatorEvents(EventEnum):
    VALIDATION_COMPLETED = 'validation_completed'


event_to_attr = {
    EvaluatorEvents.VALIDATION_COMPLETED: 'validation_completed'
}

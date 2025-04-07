import copy

class UndoRedoManager:
    """
    Manages the history of annotations for undo and redo operations.
    Stores annotation states as snapshots for efficient history tracking.
    """
    def __init__(self, max_history=50):
        self.history = []
        self.current_index = -1
        self.max_history = max_history
    
    def push_state(self, state):
        """
        Push a new state to the history.
        
        Args:
            state: Current annotation state to save
        """
        # Make a deep copy to avoid reference issues
        state_copy = copy.deepcopy(state)
        
        # If we're not at the end of the history, truncate
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        
        # Add new state
        self.history.append(state_copy)
        self.current_index = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_index -= 1
    
    def undo(self):
        """
        Undo to the previous state.
        
        Returns:
            state: Previous state, or None if no history
        """
        if self.current_index > 0:
            self.current_index -= 1
            return copy.deepcopy(self.history[self.current_index])
        return None
    
    def redo(self):
        """
        Redo to the next state.
        
        Returns:
            state: Next state, or None if at the end of history
        """
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return copy.deepcopy(self.history[self.current_index])
        return None
    
    def current_state(self):
        """
        Get the current state.
        
        Returns:
            state: Current state, or None if no history
        """
        if self.current_index >= 0 and self.current_index < len(self.history):
            return copy.deepcopy(self.history[self.current_index])
        return None
    
    def can_undo(self):
        """Check if undo operation is available"""
        return self.current_index > 0
    
    def can_redo(self):
        """Check if redo operation is available"""
        return self.current_index < len(self.history) - 1
    
    def clear(self):
        """Clear history"""
        self.history = []
        self.current_index = -1
"""
API service interface for Habitat Evolution.

This module defines the interface for API services in the Habitat Evolution system,
providing a consistent approach to external API interactions across the application.
"""

from typing import Protocol, Dict, List, Any, Optional, Union
from abc import abstractmethod

from src.habitat_evolution.infrastructure.interfaces.service_interface import ServiceInterface


class APIServiceInterface(ServiceInterface, Protocol):
    """
    Interface for API services in Habitat Evolution.
    
    API services provide a consistent approach to external API interactions,
    abstracting the details of HTTP requests and response handling.
    This supports the pattern evolution and co-evolution principles of Habitat
    by enabling flexible API integration while maintaining a consistent interface.
    """
    
    @abstractmethod
    def request(self, method: str, endpoint: str, 
               params: Optional[Dict[str, Any]] = None,
               data: Optional[Any] = None,
               headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: The HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: The API endpoint
            params: Optional query parameters
            data: Optional request data
            headers: Optional request headers
            
        Returns:
            The API response
        """
        ...
        
    @abstractmethod
    def get(self, endpoint: str, 
           params: Optional[Dict[str, Any]] = None,
           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: The API endpoint
            params: Optional query parameters
            headers: Optional request headers
            
        Returns:
            The API response
        """
        ...
        
    @abstractmethod
    def post(self, endpoint: str, 
            data: Optional[Any] = None,
            params: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: The API endpoint
            data: Optional request data
            params: Optional query parameters
            headers: Optional request headers
            
        Returns:
            The API response
        """
        ...
        
    @abstractmethod
    def put(self, endpoint: str, 
           data: Optional[Any] = None,
           params: Optional[Dict[str, Any]] = None,
           headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a PUT request.
        
        Args:
            endpoint: The API endpoint
            data: Optional request data
            params: Optional query parameters
            headers: Optional request headers
            
        Returns:
            The API response
        """
        ...
        
    @abstractmethod
    def delete(self, endpoint: str, 
              params: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: The API endpoint
            params: Optional query parameters
            headers: Optional request headers
            
        Returns:
            The API response
        """
        ...
        
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the API service.
        
        Args:
            config: Configuration for the API service
        """
        ...
        
    @abstractmethod
    def set_auth_token(self, token: str) -> None:
        """
        Set the authentication token.
        
        Args:
            token: The authentication token
        """
        ...
        
    @abstractmethod
    def refresh_auth_token(self) -> str:
        """
        Refresh the authentication token.
        
        Returns:
            The new authentication token
        """
        ...

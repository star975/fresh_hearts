describe('Fresh Hearts App', () => {
    const baseUrl = 'http://localhost:8501';
  
    beforeEach(() => {
      cy.visit(baseUrl);
    });
  
    it('should display the login page initially', () => {
      cy.contains('Login to Fresh Hearts').should('exist');
    });
  
    it('should allow user to login and see welcome message', () => {
      // Customize selectors based on your actual form structure
      cy.get('input[type="text"]').first().type('testuser');
      cy.get('input[type="password"]').first().type('password123');
      cy.contains('Login').click();
  
      // Wait for rerender, then check welcome message
      cy.contains('Welcome to ❤️ Fresh Hearts', { timeout: 5000 }).should('be.visible');
      cy.contains('A healthy tomorrow matters').should('be.visible');
    });
  
    it('should logout and return to login screen', () => {
      // Simulate login
      cy.get('input[type="text"]').first().type('testuser');
      cy.get('input[type="password"]').first().type('password123');
      cy.contains('Login').click();
  
      // Wait then logout
      cy.contains('Logout', { timeout: 5000 }).click();
      cy.contains('Login to Fresh Hearts').should('exist');
    });
  });
  
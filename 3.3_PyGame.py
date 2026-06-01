import pygame # Pygame einbinden

pygame.init() # Pygame initialisieren
screen = pygame.display.set_mode((800, 500))
endlos=True # endlos währt am längsten

while endlos==True:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        endlos = False
        screen.fill((0, 0, 0)) # Bildschirm löschen
        # Zeichne ein 20x20 Kästchen mit Grauverlauf
        for x in range(0,20): 
            for y in range(0,20):
                col = x*10
                pygame.draw.rect(screen, (col, col, col), 
                        pygame.Rect(x*10, y*10, 10, 10))
        pygame.display.flip() # Zeige das Gezeichnete an
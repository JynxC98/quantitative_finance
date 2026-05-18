/**
 * @brief This script is used to demonstrate the example usecase of hierarchial
 * inheritance.
 */

/**
 * @brief This script aggregates all the inheritance concepts into one file
 */
#include <iostream>
#include <string>

class Animal
{
private:
    int num_legs;
    bool isCarnivore;
    std::string environment;

public:
    std::string name;

    Animal(std::string n, int legs, bool carnivore, std::string env)
        : name(n), num_legs(legs), isCarnivore(carnivore), environment(env) {}

protected:
    int getNumLegs() const { return num_legs; }
    bool getIsCarnivore() const { return isCarnivore; }
    std::string getEnvironment() const { return environment; }

public:
    virtual void makeSound() const
    {
        std::cout << name << " makes a generic animal sound" << std::endl;
    }

    void displayInfo() const
    {
        std::cout << "Name: " << name << std::endl;
        std::cout << "Legs: " << num_legs << std::endl;
        std::cout << "Carnivore: " << (isCarnivore ? "Yes" : "No") << std::endl;
        std::cout << "Environment: " << environment << std::endl;
    }

    virtual ~Animal() {}
};

class Bird : public Animal
{
protected:
    double wingspan;

public:
    Bird(std::string n, double wingspan, int legs = 2, bool carnivore = false, std::string env = "Air")
        : Animal(n, legs, carnivore, env), wingspan(wingspan) {}

    // Virtual function - can be overridden
    virtual void fly() const
    {
        std::cout << name << " is flying with " << wingspan << "m wingspan!" << std::endl;
    }

    void makeSound() const override
    {
        std::cout << name << " chirps!" << std::endl;
    }
};

class Eagle : public Bird
{
private:
    double eyesightPower;

public:
    Eagle(std::string n, double wingspan, double eyesight)
        : Bird(n, wingspan, 2, true, "Mountains"), eyesightPower(eyesight) {}

    void makeSound() const override
    {
        std::cout << name << " the eagle screeches!" << std::endl;
    }

    void fly() const override
    { // ✅ Works - Bird::fly() is virtual
        std::cout << name << " soars high with " << wingspan << "m wings!" << std::endl;
    }

    void hunt() const
    {
        std::cout << name << " is hunting with " << eyesightPower << "x vision!" << std::endl;
    }
};

class Penguin : public Bird
{
public:
    Penguin(std::string n, double wingspan)
        : Bird(n, wingspan, 2, false, "Antarctic") {}

    void makeSound() const override
    {
        std::cout << name << " honks!" << std::endl;
    }

    void fly() const override
    { // ✅ Now this works!
        std::cout << name << " cannot fly, but swims very well!" << std::endl;
    }

    void swim() const
    {
        std::cout << name << " is swimming in freezing water!" << std::endl;
    }
};

int main()
{
    // Demonstrating polymorphism with virtual functions
    std::vector<Bird *> birds;

    Eagle eagle("Freedom", 2.5, 8.0);
    Penguin penguin("Pingu", 0.5);

    birds.push_back(&eagle);
    birds.push_back(&penguin);

    std::cout << "=== Polymorphic Bird Behaviors ===" << std::endl;
    for (Bird *bird : birds)
    {
        bird->fly(); // Calls appropriate version based on actual object type
        bird->makeSound();
        std::cout << std::endl;
    }

    // Specific behaviors
    eagle.hunt();
    penguin.swim();

    return 0;
}
/**
 * This script is used to demonstrate the functioning of single level inheritence
 * using cpp
 */

/**
 * @brief This script is used to demonstrate the example usecase of inheritance
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

    // Initializing a constructor
    Animal(std::string n, int legs, bool carnivore, std::string env)
        : name(n), num_legs(legs), isCarnivore(carnivore), environment(env) {}

protected:
    int getNumLegs() const { return num_legs; }
    bool getIsCarnivore() const { return isCarnivore; }
    std::string getEnvironment() const { return environment; }

public:
    // Public methods available to all
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

    virtual ~Animal() {} // Virtual destructor for proper cleanup
};

// Dog inherits from Animal
class Dog : public Animal
{
private:
    std::string breed;

public:
    Dog(std::string n, std::string b, int legs = 4, bool carnivore = false, std::string env = "Domestic")
        : Animal(n, legs, carnivore, env), breed(b) {}

    // Override base class method
    void makeSound() const override
    {
        std::cout << name << " the " << breed << " says: Woof! Woof!" << std::endl;
    }

    void wagTail() const
    {
        std::cout << name << " is wagging tail!" << std::endl;
    }
};
